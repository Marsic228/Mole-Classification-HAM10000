def summarize_experiment(image_records, config, run):
    first_image = None
    label_counts = {}

    for i in image_records:
        label = i.label

        if label not in label_counts:
            label_counts[label] = 0

        label_counts[label] += 1

    if image_records:
        first_image = image_records[0].to_dict()

    return {
        "image_count": len(image_records),
        "dataset_config": config.to_dict(),
        "experiment_run": run.to_dict(),
        "first_image": first_image,
        "label_counts": label_counts
    }