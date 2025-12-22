# %% --------------------------------------

import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    text_input = mo.ui.text(placeholder="My name is ...", debounce=False)
    return (text_input,)


@app.cell
def _(mo, text_input):
    mo.md(f"My name is {text_input}")
    return


@app.cell
def _(text_input):
    text_input.value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
