### Parastesh Talebi ###
import turtle
t  =turtle.Turtle()
t.shape('turtle')
t.color('green')
t.width(3)
for i in range(8):
    for j in range(4):
        t.forward(200)
        t.left(90)
    t.left(45)
turtle.done()