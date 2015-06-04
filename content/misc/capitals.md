Title: Playing Capitals with OpenCV and Python
Date: 2015-06-04 17:05
Tags: computer math, hexagons, capitals
Summary: On Monday evening I had dinner with a friend who showed me the game [Capitals](https://itunes.apple.com/us/app/capitals-free-word-battle/id968456900) and suggested it might be interesting to play it programmatically. tl;dr: I spent the last 18-20 working hours doing that [(Github link)](https://github.com/iank/capitals-solver)
Status: draft

On Monday evening I had dinner with my friend Alysia Promislow, who showed me the game [Capitals](https://itunes.apple.com/us/app/capitals-free-word-battle/id968456900) and suggested it might be interesting to play it programmatically.

I spent the last 18-20 working hours doing that. [(Github link with code and example images)](https://github.com/iank/capitals-solver). Hopefully it's redundant to mention that I've done this because it let me geek out on a goofy computational problem, not because I'm interested in cheating at a phone game[^game].

90% of getting anything done is knowing things exist, and the first two things I thought about after seeing the game is:

- The low-complexity shapes and clear separation make it amenable to some simple computer vision techniques [that I have employed before](http://iank.org/rmbc.html)
- Hexagonal coordinate systems are a thing

### Decoding game state from a screenshot

I spent Tuesday afternoon writing a Python script to decode the game state from a screenshot, using [OpenCV](http://opencv.org) and [Tesseract](https://code.google.com/p/tesseract-ocr/) OCR engine.

It takes an RGB image, such as:

[![RGB screenshot](/images/capitals_ex.png)

Next, it greyscales and runs the [Canny edge detector](http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=canny#canny) to produce:

![Edge-detected](/images/capital_ex_canny.png)

To detect and isolate hexagons, I follow a [similar approach](https://github.com/Itseez/opencv/blob/master/samples/cpp/squares.cpp) as the OpenCV example [squares.cpp](https://github.com/Itseez/opencv/blob/master/samples/cpp/squares.cpp):

- [Find contours](http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours) in the edge-detected image
- Using the [Ramer-Douglas-Peucker algorithm](http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#approxpolydp), attempt to approximate polygonal curves as lower-degree polygons
- Take all detected six-sided convex polygons having a certain minimum area
- It is also useful to check for approximately 120 degree angles, but it's not necessary for this application.

Taking only the contours which meet the criteria, we have:

![Isolated contours](/images/capital_ex_contour_mask.png)

Iterating through these I use a series of increasingly-dubious heuristics to classify the hexagons:

- Take the average RGB value of each hexagon:
    - Hexagons that are much more red than blue belong to red player
    - Hexagons that are much more blue than red belong to blue player
- Hexagons that are mostly white belong to neither, and I determine the letter by passing the masked image to [Tesseract](https://code.google.com/p/tesseract-ocr/)
- Rather than searching for the icon that denotes the capital, I count white pixels in each red and blue hexagon. e.g., a hexagon that is mostly red but has a significant white space is the red capital.

Now I can find possible words, but only some of them are useful. Also, there are typically thousands of candidates. In order to consider word connectedness, I find the centroids of each hexagon and estimate their position on a hexagonal grid. I've encountered hexagonal coordinate systems in [wireless communications](http://en.wikipedia.org/wiki/File:CellTowersAtCorners.gif), but I didn't know about the [Q*bert equivalence](http://keekerdc.com/2011/03/hexagon-grids-coordinate-systems-and-distance-calculations/). (There's also a great [animation here](http://www.redblobgames.com/grids/hexagons/), search for "convert to cube coordinates").

Here is [my derivation](/images/hex_derivation.jpg) for the mapping from rectangular to hexagonal coordinates, given the hexagonal side length (estimated from detected contours) and an origin (arbitrarily chosen, it only needs to be relatively consistent). Also there is a spacing between hexagons in this game, which I have denoted 'b'.

### Finding useful moves from game state

On Wednesday afternoon I wrote code to score candidate words. It is trivial to find all possible words, given a list of letters and a dictionary. However as mentioned above, a word's length is not the most useful indicator of its fitness as a move in the game. For a played word, letters that are connected to the player's territory will become a part of it. Isolated letters can be used to construct a word, but will not become part of player territory. Also, if connected tiles in a word are adjacent to enemy territory, the opposing player will lose that territory. Finally, it is frequently possible to create the same word using differently-located tiles, and this has a strategic impact (so words are not unique, combinations of tiles are).

I score candidate word choices by counting the length of the word, number of tiles it will add to player territory, number of tiles it will remove from enemy territory, and whether one of those tiles is the enemy capital (thus granting the player an extra turn, usually allowing a win).[^vars]

I determine connectedness of a candidate word by considering the list of all currently-owned tiles. For each of these, I check each of the six adjacent tiles. If it is part of the candidate word, add to the list of owned tiles, and note that I've visited it. Iterate until I am done with the list of owned tiles. The result is a list of all connected tiles in the candidate word, which I then use to check enemy player adjacency. For each candidate word a vector of these score variables is generated, and I do some rudimentary sorting to produce a suggestion, such as:

    word:    retrenching, territory gain   10, enemy territory loss    6
    word:    incremented, territory gain   10, enemy territory loss    5
    word:     converting, territory gain   10, enemy territory loss    5
    word:     converting, territory gain   10, enemy territory loss    5
    word:     monteverdi, territory gain   10, enemy territory loss    5
    word:     reentering, territory gain    9, enemy territory loss    6
    word:    retrenching, territory gain    9, enemy territory loss    6
    word:    retrenching, territory gain    9, enemy territory loss    6
    word:      comintern, territory gain    9, enemy territory loss    5

![Suggestion](/images/capitals_ex_suggestion.png)

[^game]: I'm terrible at this game, though. Scrabble too.

[^vars]: There are other variables to consider such as protecting one's own capital, avoiding being the player to bridge the gap, and positioning (ie gaining tiles in the center is more important than gaining isolated tiles which have to path to the enemy player). My thought is to construct a feature vector with these variables, weight them by some vector $\alpha$, and use a genetic algorithm to pit several of these automated players against each other in order to learn $\alpha$. I may never get around to it.
