# TODO

## Binary Border Image

I had problems getting opencv to find the rectangle. It doesn't want to track the inner edge.

As an alternative we coudl use some kind of flood fill, where you start inside the rectangle and go to the bottom right corner. The algorithm is to just go down, right. If you get stuck follow left until you can go down right. Should be good enough but might not work in edge cases, where you get stuck in upper right of mid-cross. Do this for all corners and this will find us the rectangle.