Point(1) = {0, 0, 0, 1.0};
Point(2) = {60, 0, 0, 1.0};
Point(3) = {60, 32, 0, 1.0};
Point(4) = {0, 32, 0, 1.0};
Point(5) = {16, 16, 0, 1.0};
Point(6) = {17.5, 16, 0, 1.0};
Point(7) = {14.5, 16, 0, 1.0};
Circle(5) = {7, 5, 6};
Circle(6) = {6, 5, 7};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {4, 1, 2, 3};
Physical Line("bottom") = {1};
Physical Line("right") = {2};
Physical Line("top") = {3};
Physical Line("left") = {4};
Physical Line("circle") = {6, 5};
Line Loop(6) = {6, 5};
Plane Surface(1) = {5, 6};
Physical Surface(6) = {1};