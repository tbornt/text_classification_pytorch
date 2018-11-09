def check_fields(required_fields, session):
    for field in required_fields:
        if field not in session:
            raise Exception('%s should be configured in IO session' % field)


def print_progress(text):
    print("=====%s=====" % text)
