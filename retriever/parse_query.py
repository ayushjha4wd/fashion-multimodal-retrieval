def parse_query(query):
    query = query.lower()
    attrs = {
        "color": [],
        "clothing": [],
        "context": [],
        "style": []
    }

    vocab = {
        "color": ["red", "blue", "yellow", "black", "white"],
        "clothing": ["shirt", "t-shirt", "jacket", "blazer", "raincoat"],
        "context": ["office", "park", "street", "home"],
        "style": ["formal", "casual"]
    }

    for k in vocab:
        for v in vocab[k]:
            if v in query:
                attrs[k].append(v)

    return attrs
