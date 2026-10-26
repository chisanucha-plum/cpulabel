import sys
import logging
from schemas import ConfigLoader, ProcessingStats
from utils import (
    setup_dataset_structure, get_image_files, 
    save_coco_json, save_yolo_yaml, create_coco_categories
)
from pipeline import ImageProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(augment: bool = False, review: bool = False) -> None:
    """Run auto-labeling pipeline."""
    try:
        config = ConfigLoader.load("configuration.json")
        setup_dataset_structure(config)

        image_files = get_image_files(config.input_folder)
        if not image_files:
            logger.error(f"No images found in {config.input_folder}")
            return

        processor = ImageProcessor(config, augment=augment, review=review)
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": create_coco_categories(config),
        }

        img_id = 1
        ann_id = 1
        counts = {"auto": 0, "confident": 0, "uncertain": 0, "rejected": 0}

        for img_path in image_files:
            for res, img in processor.process_image(img_path, img_id):
                if res.category == "rejected":
                    counts["rejected"] += 1
                    logger.debug(f"Rejected: {res.file_name}")
                    continue

                coco_data["images"].append({
                    "id": res.image_id,
                    "file_name": res.file_name,
                    "width": res.width,
                    "height": res.height,
                })
                anns, ann_id = processor.save_results(res, img, ann_id, res.category)
                coco_data["annotations"].extend(anns)
                counts[res.category] += 1
                _update_stats(processor.stats, res, config)
                img_id = res.image_id + 1

        save_coco_json(coco_data, config)
        save_yolo_yaml(config)
        _print_summary(config, processor.stats, len(image_files), counts, review)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def _update_stats(stats: ProcessingStats, result, config) -> None:
    stats.processed_images += 1
    stats.total_detections += len(result.detections)
    for det in result.detections:
        name = config.classes[det.class_id]
        stats.detections_by_class[name] = stats.detections_by_class.get(name, 0) + 1


def _print_summary(
    config,
    stats: ProcessingStats,
    total_images: int,
    counts: dict = None,
    review: bool = False,
) -> None:
    logger.info("=" * 60)
    logger.info("✓ Complete!")
    logger.info(f"  Images processed: {stats.processed_images}/{total_images}")
    logger.info(f"  Annotations: {stats.total_detections}")

    if review and counts:
        logger.info("\n  Categories:")
        logger.info(f"    Confident: {counts['confident']}")
        logger.info(f"    Uncertain: {counts['uncertain']}")
        logger.info(f"    Rejected: {counts['rejected']}")

    if stats.detections_by_class:
        logger.info("\n  By class:")
        for name, count in sorted(stats.detections_by_class.items()):
            logger.info(f"    {name}: {count}")

    logger.info(f"\n📁 Output: {config.output_folder}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    if "--verbose" in sys.argv:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        augment="--augment" in sys.argv,
        review="--review" in sys.argv,
    )
