from functools import singledispatch


@singledispatch
def ignore_error(classifiers) -> None:
    try:
        classifiers()
    except Exception as e:
        with open("error.log", "a") as f:
            f.write(f"{e}\n")


@ignore_error.register
def _(classifiers: list)->None:
    for classifier in classifiers:
        try:
            classifier()
        except Exception as e:
            with open("error.log", "a") as f:
                f.write(f"{e}\n")
