from grandmaster import task

task.load_model(
    task_name="zero-shot-image-classification",
    model_name="AUTO",
)
puppy: bytes = load_image_data_from_url(
    "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
)
task.inference(image=puppy, candidate_labels=["dog", "cat"])
