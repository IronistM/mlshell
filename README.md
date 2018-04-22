# ML Shell

Empty shell for Machine Learning projects. To be used as a base project, and tweaked according to needs.

## Breakdown of main Classes

All logic follows a functional programming paradigm.

* `DataSet` from `lib/dataset.py`: query related logic.
The idea is to download raw datasets and keep them in memory. Other classes and scripts can then build on top of them and perform more advanced filtering, joining, etc in RAM.
* `FeatureConstructor` from `lib/feature_constructor.py`: more advanced data manipulation that are specific to the dataset.
Joining, merging, filtering, or other operations that make sense specifically for the current dataset (e.g. business related data transforms).
* `FeatureTransformer` from `lib/feature_transformer.py`: generic data transforms.
Generic transformations that can be applied to data types (e.g. categorical features) independently of business logic: scaling, encoding, normalizing, etc.
* `BinaryClassifier` from `models/binary_classifier.py`: augmented `sklearn` object, with built in metrics, scoring, saving, etc.


