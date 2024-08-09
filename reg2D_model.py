# Function to generate 2D scatter plot for a regression model and a dataset
#
################################################################################
# Dr. Walter F. de Azevedo, Jr.                                                #
# https://github.com/azevedolab                                                #
# July 20, 2024                                                                #
################################################################################
#
# Import section
import matplotlib.pyplot as plt

# Define plotting() function
def plotting(title_str,
                X,y,plt_str_1,plt_axis_array,
                X_new,y_predict,plt_str_2,label_str,x_label_str,y_label_str,
                legend_loc,grid_boolean,output_plot,dpi_plot):
    """Function to generate a 2D plot of a dataset and its regression model"""

    # Plotting
    plt.title(title_str)
    plt.plot(X, y,plt_str_1)                               # Experimental points
    plt.plot(X_new,y_predict,plt_str_2,label=label_str)    # Predictions
    plt.axis(plt_axis_array)                               # Limits of axis
    plt.xlabel(x_label_str)                                # X label
    plt.ylabel(y_label_str)                                # y label
    plt.legend(loc=legend_loc)                             # Legend location
    plt.grid(visible = grid_boolean)                       # Add grid to plot
    plt.show()                                             # Show plot
    plt.savefig(output_plot,dpi=dpi_plot)                  # Save plot