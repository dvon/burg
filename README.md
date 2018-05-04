## Background

The goal of this project is to significantly speed up data collection
from scanned images of handwritten census forms (from approximately
100 years ago).
OCR software works well for printed text, but the cursive
handwriting in these forms presents a much
more challenging problem. Our intent is not to fully
automate the data collection process, but to guide a human user in
a way that makes their work more efficient.

Rather than attempting
to develop better OCR software, with the goal of automatically
recognizing letters within the handwritten text, we intend to use
existing image processing / computer vision techniques designed to
group together similar images. Our software would not recognize,
for example, that the images in one group represented the
handwritten word “yes” and in another group the word “no.” The
software would simply categorize images into groups and present
representative images from each group to a human user, who would
interpret the images’ meaning, which could be automatically applied
to the other images from the group.

## So Far...

The example images, with explanations for where they come from,
are meant to show what's been accompished so far.  The current
version is written in Python 3 and uses OpenCV 3.

[example.jpg](example.jpg)

The original scanned census form.

[example\_corners.jpg](example\_corners.jpg)

The census form, with corners of the main table marked.  Uses
`fast_corners` function in crop.py.  Original `corners` function
searches top left, top right, bottom right, and then bottom left
portions of the original image for corners; `fast_corners` version
scales the image down and searches the scaled down version.  Both
functions rely on `cv2.matchTemplate` to find the corners.

[example\_cropped.jpg](example\_cropped.jpg)

After corners are found, `deskew` function (in crop.py) uses
`cv2.findHomography` to get the transform needed to make the
table in the image a true rectangle, and to align it vertically
and horizontally.  `cv2.warpPerspective` is used to apply the
transform.  Then the `crop` function is used to get a view
(i.e., a numpy view, rather than a copy) of the image data limited
to the table and a small margin around the outside.

[example\_lines.jpg](example\_lines.jpg)

`lines` function, in cells.py, is used to find the horizontal and
vertical boundary lines in the table.  `filter_lines` removes
redundant lines, found one pixel apart, that represent wider
boundary lines in the table.  `draw_lines` is used to produce
an image like this one, for testing.

Line detection is done using `cv2.HoughLines` on a highly
processed version of the image; how well it works is very
sensitive to several parameters defined at the
beginning of cells.py.  I have them adjusted for the example
scan; probably they won't be quite right for others.  It may
eventually be necessary to build something to
automatically adjust these parameters based on the number of
lines we expect to find in the table.

[example\_cells.jpg](example\_cells.jpg)

Once the table boundary lines have been found, the `cells` function
(in cells.py), determines the coordinates of corners for cells
defined by those boundaries.  `adjust_borders` uses
`cv2.matchTemplate` to search cell images for horizontal and
vertical border lines (maybe template matching isn't the best way
to do this?) along the outside edges, and adjusts corner
coordinates based on what it finds.  This wouldn't be necessary
if the lines in the table were perfectly straight and evenly
spread out, but they are far enough off in some places that this
extra step helps.

The `erase_borders` function works in a way very similar to
`adjust_borders`, except that it colors the border lines white.
It then processes (using close transformations with a horizontally
oriented kernel and then a vertically oriented kernel) the portion
of the cell image where borders
were erased in a way that "spreads" the erasing a little bit
further out, to catch missed border bits, and also fills in a
little bit of darker color where handwriting lines crossed the
original border.  (This part works well enough to use it, but
there's definitely potential for improvement.)

Finally, `draw_cells` creates a version of the image with the
borders erased and with cells highlighted by colored rectangles.
(This is to test the effectiveness of the cell finding and
border erasing functions, particularly as line detection parameters
are modified and it's important to see the effect.)

[example\_compare.jpg](example\_compare.jpg)

This image represents a first attempt to group cells in a column
by similarity.  It uses a simple template matching scheme
(`cv2.matchTemplate` again), with very little preprocessing.
Whether two cells are considered similar enough to group together
is just based on a threshold value for how good a match could
be found between the two cells.  The image shows (highlighted
with colored rectangles) cells that supposedly match the first
cell in each column.  (This was produced by an earlier version of
compare.py.)

[example\_compare\_closest.jpg](example\_compare\_closest.jpg)

This image represents a second attempt to group cells by similarity,
more conservative than the first attempt.  Template matching is
used to get a similarity score, but instead of grouping together
all matches above a threshold, a graph is created linking per-cell
nodes to their best match.  Connected components within the graph
are considered similar enough to group together.  (This image
was produced by the current version of compare.py.)

Within in a column, there's a different highlight color for
each of the first 8 groups found.  In the "RELATION" column, for
example, cells grouped with the first cell are tan, cells grouped
with the second and third are blue, with the fourth are red, etc.

[example\_compare\_t\_closest.jpg](example\_compare\_t\_closest.jpg)

This image represents a third attempt to group cells, very much
like the second attempt except that, instead of considering just
the closest match for each cell, it considers the top T closest
matches.  Also, cells whose average color is very close to white
are assumed to be blank and ignored.
