import sys
import logging
from pathlib import Path

from schemas import ConfigLoader, ProcessingStats
from utils_module import (
    setup_dataset_structure, get_image_files, 
    save_coco_json, save_yolo_yaml, create_coco_categories
)
from processors import ImageProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(augment: bool = False):
    """Run auto-labeling pipeline
    
    Args:
        augment: Enable data augmentation (5x images)
    """
    try:
        # Load configuration
        config = ConfigLoader.load("configuration.yaml")
        
        # Setup output structure
        setup_dataset_structure(config)
        
        # Get input images
        image_files = get_image_files(config.input_folder)
        if not image_files:
            logger.error(f"No images found in {config.input_folder}")
            return
        
        # Initialize processor
        processor = ImageProcessor(config, augment=augment)
        
        # Prepare COCO dataset
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": create_coco_categories(config)
        }
        
        annotation_id = 1
        
        # Process each image
        for image_id, image_path in enumerate(image_files, start=1):
            result_list = processor.process_image(image_path, image_id)
            
            for result_data in result_list:
                if result_data is None:
                    continue
                
                result, image_source = result_data
                
                # Add to COCO dataset
                coco_data["images"].append({
                    "id": result.image_id,
                    "file_name": result.file_name,
                    "width": result.width,
                    "height": result.height
                })
                
                # Save results
                coco_annotations, annotation_id = processor.save_results(
                    result, image_source, annotation_id
                )
                coco_data["annotations"].extend(coco_annotations)
                
                # Update statistics
                _update_stats(processor.stats, result, config)
        
        # Save outputs
        save_coco_json(coco_data, config)
        save_yolo_yaml(config)
        
        # Print summary
        _print_summary(config, processor.stats, len(image_files))
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def _update_stats(stats: ProcessingStats, result, config):
    """Update processing statistics"""
    stats.processed_images += 1
    stats.total_detections += len(result.detections)
    
    for det in result.detections:
        class_name = config.classes[det.class_id]
        stats.detections_by_class[class_name] = \
            stats.detections_by_class.get(class_name, 0) + 1

def _print_summary(config, stats: ProcessingStats, total_images: int):
    """Print processing summary"""
    logger.info(f"{'='*60}")
    logger.info(f"✓ Complete!")
    logger.info(f"  Images: {stats.processed_images}/{total_images}")
    logger.info(f"  Annotations: {stats.total_detections}")
    
    if stats.detections_by_class:
        logger.info(f"\n  By class:")
        for class_name, count in sorted(stats.detections_by_class.items()):
            logger.info(f"    {class_name}: {count}")
    
    logger.info(f"\n Output: {config.output_folder}/")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    augment = "--augment" in sys.argv
    verbose = "--verbose" in sys.argv
    
    # Set log level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    run_pipeline(augment=augment)
