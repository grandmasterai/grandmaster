from grandmaster import task

model = task(name="zero-shot-image-classification")

prediction = model.prediction(
    image="http://localhost:3000/ex/puppy.jpeg", candidate_labels=["dog", "cat"]
)

print(prediction)
