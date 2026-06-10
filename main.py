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

def run_pipeline(augment: bool = False, review: bool = False):
    """Run auto-labeling pipeline
    
    Args:
        augment: Enable data augmentation (5x images)
        review: Enable human-in-the-loop review (auto-sorts by confidence)
    """
    try:
        # Load configuration
        config = ConfigLoader.load("configuration.json")
        
        # Setup output structure
        setup_dataset_structure(config)
        _setup_category_folders(config)
        
        # Get input images
        image_files = get_image_files(config.input_folder)
        if not image_files:
            logger.error(f"No images found in {config.input_folder}")
            return
        
        # Initialize processor
        processor = ImageProcessor(config, augment=augment, review=review)
        
        # Prepare COCO dataset
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": create_coco_categories(config)
        }
        
        annotation_id = 1
        categories_count = {"auto": 0, "confident": 0, "uncertain": 0, "rejected": 0}
        
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
                
                # Save results with category organization
                if result.category != "rejected":
                    coco_annotations, annotation_id = processor.save_results(
                        result, image_source, annotation_id, result.category
                    )
                    coco_data["annotations"].extend(coco_annotations)
                    categories_count[result.category] += 1
                    
                    # Update statistics
                    _update_stats(processor.stats, result, config)
                else:
                    categories_count["rejected"] += 1
                    logger.debug(f"Rejected: {result.file_name}")
        
        # Save outputs 
        save_coco_json(coco_data, config)
        save_yolo_yaml(config)
        
        # Print summary
        _print_summary(config, processor.stats, len(image_files), categories_count, review)
    
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

def _print_summary(config, stats: ProcessingStats, total_images: int, categories_count: dict = None, review: bool = False):
    """Print processing summary"""
    logger.info(f"{'='*60}")
    logger.info(f"✓ Complete!")
    logger.info(f"  Images processed: {stats.processed_images}/{total_images}")
    logger.info(f"  Annotations: {stats.total_detections}")
    
    if review and categories_count:
        logger.info(f"\n  Categories:")
        logger.info(f"    Confident: {categories_count['confident']}")
        logger.info(f"    Uncertain: {categories_count['uncertain']}")
        logger.info(f"    Rejected: {categories_count['rejected']}")
    
    if stats.detections_by_class:
        logger.info(f"\n  By class:")
        for class_name, count in sorted(stats.detections_by_class.items()):
            logger.info(f"    {class_name}: {count}")
    
    logger.info(f"\n📁 Output: {config.output_folder}/")
    logger.info(f"{'='*60}")

def _setup_category_folders(config):
    """Create folders for different confidence categories"""
    base = Path(config.output_folder)
    for category in ["confident", "uncertain"]:
        for subfolder in ["images", "labels"]:
            folder = base / "yolo" / category / subfolder
            folder.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    augment = "--augment" in sys.argv
    review = "--review" in sys.argv
    verbose = "--verbose" in sys.argv
    
    # Set log level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    run_pipeline(augment=augment, review=review)
