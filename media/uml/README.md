# UML diagrams for slides

Using `pyreverse` (comes with `pylint`), generate the initial `.dot` file to work with:

```console
$ pyreverse -p data_morph --ignore factory.py src/data_morph/shapes/
```

Edit the file to your liking (likely trimming it down to fit). Then, convert it to an image file:

```console
$ dot -T png input.dot > output.dot
$ dot -T svg input.dot > output.svg
```
