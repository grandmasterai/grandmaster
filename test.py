from grandmaster.Example import get_examples, example_path_to_url


examples = get_examples({"dataType": "IMAGE"})
for e in examples:
    # print(e)
    print(example_path_to_url(e))
