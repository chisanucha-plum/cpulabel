from schemas import ConfigLoader, ProcessingStats
from utils_module import setup_dataset_structure, get_image_files, save_coco_json, save_yolo_yaml, create_coco_categories
from processors import ImageProcessor

def run_pipeline():
    """Run auto-labeling pipeline"""
    
    # Load configuration
    config = ConfigLoader.load("configuration.yaml")
    
    # Setup
    setup_dataset_structure(config)
    image_files = get_image_files(config.input_folder)
    
    if not image_files:
        print(f"No images found in {config.input_folder}")
        return
    
    # Initialize processor
    processor = ImageProcessor(config)
    
    # Prepare COCO dataset
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": create_coco_categories(config)
    }
    
    annotation_id = 1
    
    # Process each image
    for image_id, image_path in enumerate(image_files, start=1):
        result_data = processor.process_image(image_path, image_id)
        
        if result_data[0] is None:
            continue
        
        result, image_source = result_data
        
        # Add image metadata
        coco_data["images"].append({
            "id": result.image_id,
            "file_name": result.file_name,
            "width": result.width,
            "height": result.height
        })
        
        # Save and get annotations
        coco_annotations, annotation_id = processor.save_results(
            result, image_source, image_path, annotation_id
        )
        coco_data["annotations"].extend(coco_annotations)
        
        # Update statistics
        _update_stats(processor.stats, result, config)
    
    # Save outputs
    save_coco_json(coco_data, config)
    save_yolo_yaml(config)
    
    # Print summary
    _print_summary(config, processor.stats, len(image_files))

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
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Images processed: {stats.processed_images}/{total_images}")
    print(f"  Total annotations: {stats.total_detections}")
    print(f"\n  Detections by class:")
    for class_name, count in sorted(stats.detections_by_class.items()):
        print(f"    - {class_name}: {count}")
    print(f"\nDataset saved to: {config.output_folder}/")
    print(f"  - YOLO: {config.output_folder}/yolo/")
    print(f"  - COCO: {config.output_folder}/coco/")
    print(f"  - Visualizations: {config.output_folder}/visualizations/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_pipeline()
