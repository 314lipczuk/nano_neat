# This is a comment
set title "My Graph Title"
set xlabel "X-axis Label"
set ylabel "Y-axis Label"

# Define plot style
set style data linespoints

# Plot data from a file
plot "actual.dat" using 1:2 title 'Actual', \
     "estimations.dat" using 1:2 title 'estimations', \
     "measurements.dat" using 1:2 with points title 'measurements', \
     "predictions.dat" using 1:2 title 'predictions'

# Alternatively, plotting mathematical functions
# plot sin(x), cos(x)