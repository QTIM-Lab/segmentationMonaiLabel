# Custom Active Learning Strategies

A custom Python class that inherits from Strategy
i.e.: from monailabel.interfaces.tasks.strategy import Strategy

## Last

Actually technically random, not last. Naming convention off due to time constraints.

Looking at last, the most confusing part would be from get_unlabeled_images() and get_labeled_images() coming from some "datastore" parameter. You could trace back that parameter, but essentially it's one of these:

[https://docs.monai.io/projects/label/en/latest/_modules/monailabel/interfaces/datastore.html#Datastore](https://docs.monai.io/projects/label/en/latest/_modules/monailabel/interfaces/datastore.html#Datastore)

And if you go to those methods, that do exactly what they say they do. And, when we get both labeled and unlabeled, we essentially get all of them. There might be an easier/better/faster way of doing this, but it worked for now. In the future, you will probably just get only unlabeled images, so no need to solve problems that may not even exist later.

Then, it simply selects a random image, and returns the id and label (which is just none) as a dict
