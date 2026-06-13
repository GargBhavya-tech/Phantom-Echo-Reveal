"""
PHANTOM-ECHO REVEAL — Real Data Evaluation Demo
Runs the unmodified PHANTOM-ECHO REVEAL pipeline on real datasets.
"""

import argparse
import logging
from pathlib import Path

# Import the original pipeline
import src.main_v2
from src.edge.sensing.real_dataset_loader import RealDepthGenerator

logger = logging.getLogger(__name__)

def run_pipeline_with_real_data(dataset_path: str, n_frames: int = 5):
    """
    Runs the existing `run_full_pipeline` from main_v2, but safely monkey-patches
    the depth generator to load real data from disk instead of using synthetic data.
    """
    logger.info(f"Preparing to run pipeline on real data from: {dataset_path}")
    
    # Keep a reference to the original so we can restore it
    OriginalGenerator = src.main_v2.SyntheticDepthGenerator
    
    class RealDataProxy:
        def __init__(self, room_dims=None, furniture=None):
            self.real_gen = RealDepthGenerator(dataset_path)
            
        def generate_walk_sequence(self, n_frames=10, start_pos=None, axis="xz"):
            return self.real_gen.generate_walk_sequence(n_frames=n_frames)
            
    # Temporarily swap the generator class
    src.main_v2.SyntheticDepthGenerator = RealDataProxy
    
    try:
        # Run the pipeline
        logger.info("Starting PHANTOM-ECHO REVEAL Pipeline...")
        result = src.main_v2.run_full_pipeline(n_frames=n_frames)
        logger.info("Pipeline completed successfully on real data.")
        return result
    finally:
        # Always restore the original class to prevent side-effects
        src.main_v2.SyntheticDepthGenerator = OriginalGenerator

def main():
    parser = argparse.ArgumentParser(description="Run pipeline on a real dataset")
    parser.add_argument("--dataset", required=True, help="Path to real dataset folder (containing color, depth, pose dirs)")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to process")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        logger.error(f"Dataset path does not exist: {args.dataset}")
        return
        
    run_pipeline_with_real_data(str(dataset_dir), n_frames=args.frames)

if __name__ == "__main__":
    main()
