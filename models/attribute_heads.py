def build_prompts(templates, values):
    prompts = []
    for v in values:
        for t in templates:
            prompts.append(t.format(v))
    return prompts
