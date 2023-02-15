from grandmaster import get_task
from grandmaster.helper import (
    get_audio,
    load_image_data_from_url,
    load_image_from_data,
    load_image_from_url,
)
from todo.mytypes import ImageQueryTypedDict, ImageZeroShotQueryTypedDict

if 1:
    # image classification
    task = get_task(
        inputs={"dataType": "IMAGE"},
        outputs={"dataType": "LABEL"},
    )
    puppy: bytes = load_image_data_from_url(
        "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
    )
    if task is not None:
        query: ImageZeroShotQueryTypedDict = {
            "image": puppy,
            "candidate_labels": ["dog", "cat"],
        }
        out = task.inference(query)
        print(out)
    else:
        print("task not found")

if 0:
    # text classification
    task = get_task(
        inputs={"dataType": "TEXT"},
        outputs={"dataType": "LABEL", "values": ["medical", "sport"]},
    )
    out = task.inference(
        "A malignant tumor in the brain is a life-threatening condition."
    )
    print(out)

if 0:
    task = get_task(
        inputs={"dataType": "image", "representation": "car"},
        outputs={"dataType": "boundingbox", "representation": "plate"},
    )
    car = load_image_from_url(
        "https://drive.google.com/uc?id=1p9wJIqRz3W50e2f_A0D8ftla8hoXz4T5"
    )
    out = task.inference(car)
    print(out)

if 0:
    task = get_task(
        inputs={
            "dataType": "image",
        },
        outputs={"dataType": "boundingbox", "representation": "face"},
    )

    obama = load_image_from_url(
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"
    )

    out = task.inference(obama)
    print(out)

if 0:
    # audio classification
    # No model exists yet for zero-shot audio classification.
    pass


if 0:
    # Text generation for scientific paper
    task = get_task(
        inputs={"dataType": "prompt", "domain": "scientific"},
        outputs={"dataType": "text"},
    )
    # get_task(Input.Prompt(domain="scientific"), Output.Generated())

    out = task.inference("The Transformer architecture [START_REF]")
    print(out)

if 0:
    # Speech to text
    task = get_task(
        inputs={"dataType": "audio"},
        outputs={"dataType": "text"},
    )
    # get_task(Input.Audio(), Output.Text())

    audio = get_audio()
    print(audio.shape)

    out = task.inference(audio)
    print(out)
