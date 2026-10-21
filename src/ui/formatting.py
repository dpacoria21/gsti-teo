"""Small presentation-only formatters."""


def number(value):
    return f"{int(value):,}".replace(",", " ")
