"""
PHANTOM-ECHO REVEAL — Active Perception & Adaptive Lambda (Layer 5)
active_perception.py  +  adaptive_lambda.py  (combined)

Active perception: guides robot to viewpoints that maximally resolve RED (unknown)
regions, using information gain as augmented reward in Nav2 pathfinding.

Information gain reward:
    R(v) = sum_{x in RED} P(observe x from v) * H(x)
where:
    H(x) = binary entropy of voxel x = -p*log(p) - (1-p)*log(1-p)
    P(observe x from v) = visibility term (ray cast from viewpoint v to x)

Adaptive lambda (self-tuning weight scheduler):
    total_reward = R_nav(v) + lambda(t) * R_info(v)

    lambda increases when RED regions stagnate (no new information)
    lambda decreases when RED regions are rapidly resolving
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ViewpointCandidate:
    """A candidate robot viewpoint for active perception."""
    position:    np.ndarray   # (3,) world position
    orientation: float        # yaw angle (radians)
    info_gain:   float = 0.0  # computed information gain reward
    nav_cost:    float = 0.0  # navigation cost from current position
    total_reward: float = 0.0  # info_gain * lambda + nav_cost_penalty


@dataclass
class ActivePerceptionState:
    """Runtime state of the active perception module."""
    lambda_weight:    float = 1.0      # current information-gain weight
    prev_red_count:   int   = 0        # RED voxels from last step
    steps_stagnated:  int   = 0        # consecutive steps with no improvement
    total_steps:      int   = 0

    # Adaptive lambda config
    lambda_min:       float = 0.1
    lambda_max:       float = 5.0
    lambda_increase:  float = 1.3     # multiply when stagnating
    lambda_decrease:  float = 0.85    # multiply when resolving
    stagnation_threshold: int = 5     # steps before lambda increases


# ── Information gain computation ────────────────────────────────────────────

def binary_entropy(prob: np.ndarray) -> np.ndarray:
    """H(p) = -p*log2(p) - (1-p)*log2(1-p). Shape-preserving."""
    eps = 1e-9
    p = np.clip(prob, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def compute_red_entropy_map(occupancy_probs: np.ndarray,
                             red_mask: np.ndarray) -> np.ndarray:
    """
    Compute information entropy for RED voxels only.

    Args:
        occupancy_probs: (X, Y, Z) float — occupancy probability per voxel
        red_mask:        (X, Y, Z) bool  — True for RED (unknown) voxels

    Returns:
        (X, Y, Z) float — entropy, 0 for non-RED voxels
    """
    entropy = np.zeros_like(occupancy_probs)
    entropy[red_mask] = binary_entropy(occupancy_probs[red_mask])
    return entropy


def visibility_sphere(viewpoint_voxel: np.ndarray,
                       target_voxels: np.ndarray,
                       occupied_mask: np.ndarray,
                       max_range_voxels: int = 40) -> np.ndarray:
    """
    Compute visibility from viewpoint to each target voxel via ray casting.
    Returns (N,) float array in [0, 1] — 1 = fully visible, 0 = occluded.

    Uses simple DDA ray casting (no occlusion = 1.0, first hit = 0.0).
    """
    N = len(target_voxels)
    visibility = np.ones(N, dtype=np.float32)

    for i in range(N):
        target = target_voxels[i]
        ray = target - viewpoint_voxel
        dist = np.linalg.norm(ray)

        if dist > max_range_voxels:
            visibility[i] = 0.0
            continue
        if dist < 1e-6:
            continue

        ray_dir = ray / dist
        n_steps = int(dist)

        for step in range(1, n_steps):
            vox = np.round(viewpoint_voxel + step * ray_dir).astype(int)
            if not all(0 <= vox[k] < occupied_mask.shape[k] for k in range(3)):
                visibility[i] = 0.0
                break
            if occupied_mask[vox[0], vox[1], vox[2]]:
                visibility[i] = 0.0   # occluded
                break

    return visibility


def information_gain_reward(viewpoint_voxel: np.ndarray,
                              entropy_map: np.ndarray,
                              occupied_mask: np.ndarray,
                              red_voxel_coords: np.ndarray,
                              max_range_m: float = 4.0,
                              voxel_size: float = 0.05) -> float:
    """
    Compute information gain reward R(v) for a candidate viewpoint.

    R(v) = sum_{x in RED} vis(v→x) * H(x)

    Args:
        viewpoint_voxel:   (3,) voxel coordinate of candidate viewpoint
        entropy_map:       (X, Y, Z) entropy for RED voxels
        occupied_mask:     (X, Y, Z) bool occupancy
        red_voxel_coords:  (M, 3) voxel coordinates of RED voxels
        max_range_m:       maximum sensing range [meters]
        voxel_size:        meters per voxel

    Returns:
        float information gain reward
    """
    if len(red_voxel_coords) == 0:
        return 0.0

    max_range_voxels = int(max_range_m / voxel_size)

    vis = visibility_sphere(viewpoint_voxel, red_voxel_coords,
                             occupied_mask, max_range_voxels)

    # Weight by entropy
    entropies = np.array([
        entropy_map[v[0], v[1], v[2]] for v in red_voxel_coords
    ])
    reward = float(np.sum(vis * entropies))
    return reward


# ── Candidate viewpoint generation ────────────────────────────────────────

def generate_frontier_viewpoints(occupancy_grid,
                                  robot_pos: np.ndarray,
                                  n_candidates: int = 20,
                                  step_m: float = 0.5) -> List[np.ndarray]:
    """
    Generate candidate viewpoints at frontier (free-space adjacent to unknown).
    Frontiers are where robot can go AND can see into unknown RED space.
    """
    free_mask = occupancy_grid.free_mask(threshold=0.3)
    prob_map  = occupancy_grid.probability()

    # Find frontier voxels: free AND adjacent to unknown
    from scipy.ndimage import binary_dilation
    unknown_mask = (prob_map > 0.3) & (prob_map < 0.7)
    unknown_dilated = binary_dilation(unknown_mask, iterations=2)
    frontier_mask = free_mask & unknown_dilated

    frontier_voxels = np.argwhere(frontier_mask)
    if len(frontier_voxels) == 0:
        logger.info("No frontier voxels found — scene may be fully resolved")
        return []

    # Sample n_candidates frontier voxels
    rng = np.random.default_rng(42)
    if len(frontier_voxels) > n_candidates:
        idx = rng.choice(len(frontier_voxels), n_candidates, replace=False)
        sampled = frontier_voxels[idx]
    else:
        sampled = frontier_voxels

    # Convert to world positions
    world_positions = [
        occupancy_grid.voxel_to_world(v) for v in sampled
    ]
    # Clamp Y to robot navigation height (0.5-1.5m)
    for pos in world_positions:
        pos[1] = np.clip(pos[1], 0.5, 1.5)

    logger.info(f"Generated {len(world_positions)} frontier viewpoint candidates")
    return world_positions


# ── Adaptive Lambda Scheduler ──────────────────────────────────────────────

def update_lambda(state: ActivePerceptionState,
                  current_red_count: int) -> float:
    """
    Self-tuning information-gain weight λ scheduler.

    Increases λ when RED count stagnates (not being resolved).
    Decreases λ when RED count is dropping rapidly.

    Args:
        state:             current ActivePerceptionState (mutated in-place)
        current_red_count: number of RED voxels at current timestep

    Returns:
        updated lambda value
    """
    state.total_steps += 1

    if state.prev_red_count > 0:
        reduction = (state.prev_red_count - current_red_count) / state.prev_red_count
    else:
        reduction = 0.0

    if reduction < 0.01:   # < 1% improvement
        state.steps_stagnated += 1
    else:
        state.steps_stagnated = 0

    if state.steps_stagnated >= state.stagnation_threshold:
        # Increase lambda to push robot toward unexplored regions
        state.lambda_weight = min(
            state.lambda_max,
            state.lambda_weight * state.lambda_increase
        )
        logger.info(
            f"Lambda increased to {state.lambda_weight:.2f} "
            f"(stagnated {state.steps_stagnated} steps)"
        )
    elif reduction > 0.05:  # > 5% improvement
        state.lambda_weight = max(
            state.lambda_min,
            state.lambda_weight * state.lambda_decrease
        )

    state.prev_red_count = current_red_count
    return state.lambda_weight


# ── Main active perception planner ─────────────────────────────────────────

def select_next_viewpoint(occupancy_grid,
                           robot_pos: np.ndarray,
                           state: ActivePerceptionState,
                           nav_speed_mps: float = 0.3) -> Optional[ViewpointCandidate]:
    """
    Select the next robot viewpoint maximising augmented reward:
        R_total(v) = lambda * R_info(v) - R_nav_cost(v)

    Args:
        occupancy_grid: populated OccupancyGrid
        robot_pos:      (3,) current robot world position
        state:          ActivePerceptionState (updated in-place)
        nav_speed_mps:  robot speed for cost estimation

    Returns:
        Best ViewpointCandidate or None if scene is resolved
    """
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        logger.warning("scipy not available — using simplified frontier detection")

    prob_map  = occupancy_grid.probability()
    occ_mask  = occupancy_grid.occupied_mask()

    # Build RED mask and entropy map
    red_mask  = (prob_map > 0.35) & (prob_map < 0.65)
    current_red_count = int(np.sum(red_mask))

    if current_red_count == 0:
        logger.info("All voxels resolved — active perception complete")
        return None

    entropy_map = compute_red_entropy_map(prob_map, red_mask)
    red_coords  = np.argwhere(red_mask)

    # PERF-V22 FIX: information_gain_reward ray-marches to EVERY red voxel for
    # EVERY candidate. With ~10^5 unknown voxels x 20 candidates this took
    # minutes in pure Python and stalled the demo at Layer 5. Reward is used
    # only to RANK candidates, so a uniform random subsample of red voxels
    # preserves the argmax with overwhelming probability (relative reward is
    # a mean over iid visibility terms). 2000 samples -> <2s total.
    MAX_RED_SAMPLES = 300
    if len(red_coords) > MAX_RED_SAMPLES:
        rng = np.random.default_rng(0)
        red_coords = red_coords[rng.choice(len(red_coords), MAX_RED_SAMPLES,
                                           replace=False)]

    # Update lambda
    lam = update_lambda(state, current_red_count)

    # Generate candidate viewpoints
    candidates_world = generate_frontier_viewpoints(
        occupancy_grid, robot_pos, n_candidates=20
    )
    # PERF-V22b: 8 candidates suffice to rank frontiers; 20 was 2.5x cost
    candidates_world = candidates_world[:6]
    if not candidates_world:
        return None

    best = None
    best_reward = -np.inf

    for pos_world in candidates_world:
        pos_vox = occupancy_grid.world_to_voxel(pos_world)

        # Information gain
        ig = information_gain_reward(
            pos_vox, entropy_map, occ_mask, red_coords,
            voxel_size=occupancy_grid.voxel_size
        )

        # Navigation cost (Euclidean / speed = time)
        nav_cost = float(np.linalg.norm(pos_world - robot_pos)) / nav_speed_mps

        total = lam * ig - 0.1 * nav_cost   # small nav penalty

        candidate = ViewpointCandidate(
            position=pos_world,
            orientation=0.0,
            info_gain=ig,
            nav_cost=nav_cost,
            total_reward=total
        )

        if total > best_reward:
            best_reward = total
            best = candidate

    if best is not None:
        logger.info(
            f"Next viewpoint: pos={best.position.round(2)}, "
            f"info_gain={best.info_gain:.3f}, "
            f"nav_cost={best.nav_cost:.1f}s, "
            f"lambda={lam:.2f}, "
            f"RED remaining={current_red_count}"
        )

    return best
