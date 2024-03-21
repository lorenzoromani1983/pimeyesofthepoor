# pimeyesofthepoor
A very poor and very simple local face recognition search engine.

You may need to search for a single face across many images.

Let's say that you find a social media post where an unknown individual pops up, and you need to find more about him.

But you only have the subject's image. No name. No references.

Then, you go to Pimeyes or other similar paid search engines, and you find a domain where that person appears. But you don't have the exact URL and reverse searching on Google/Yandex does not yield any decent result. Let's say that you manage to download many images from that web domain: your target is there, but you cannot open each image manually as it would take hours or days.

Another scenario: you have a bunch of images, downloaded from facebook, instagram, or somewhere else. And you want to find your unknown target within those images.

Pimeyes of the Poor can help you.

Running the tool is very easy.

First, you need to have Python installed.

Then, install all required dependencies:

>>> pip install -r requirements.txt

Next steps:

1) save your images (those where your target hopefully hides) in the "dataset" folder
2) save the image(s) of your unknown target in the "TARGET_FACE" folder. You may save multiple images of the unknown target if you have more than one. The more, the better.
3) run the encoding script:

>>> python encode.py

Now, the script will encode all the faces it finds in the /dataset folder and save each faces' embedding vector in the dataset.txt file.

When it's done, do this:

>>> python search.py

Now, the script will a) encode the face(s) of your unknown target (the faces you saved in the /TARGET_FACE folder) in a single face embedding.
and b) it will compute the similarity between each of the faces found in the /dataset folder and your unknown target's face embedding.

Eventually, the script will output a nifty table where the top images will be those more similar to your unknown guy. The higher the score, the more likely you found a match.
Usually, scores above 0.65 mean that you may have found a match.

NOTE: both in the "encode.py" and "search.py" scripts change the encoding engine to "hog" instead of "cnn" if you do not have a GPU.
NOTE/2 - when you have multiple images of an unkwown target face, the script will compute multiple embedding vectors, and then it will compute a "mean" vector - this is why having multiple images of an unknown target improves accuracy.

Enjoy!


