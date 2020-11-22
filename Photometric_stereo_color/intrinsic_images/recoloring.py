

import numpy as np
from PIL import Image

ball = Image.open('/home/isualice/UvA/lab1/intrinsic_images/ball.png')


width, height = ball.size[0], ball.size[1]

# Process every pixel
for x in range(width):
    for y in range(height):
        current_color = ball.getpixel( (x,y) )
        for i in current_color:
        # Only replace with the new pixel tuple if it is not pure black
           if i != 0:
            new_color = (0,255,0)

            ball.putpixel( (x,y), new_color)


ball.save('recolored.jpg')
ball.show()