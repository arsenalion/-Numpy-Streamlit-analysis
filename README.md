## Downloading the Dataset

You will need to download some prerequisite packages in order to run all the code below. Let's install them!




%%capture
!pip install numpy pandas streamlit gdown currencyconverter

import numpy as np

# For readability purposes, we will disable scientific notation for numbers
np.set_printoptions(suppress=True)

Taking a look at the `import` statements below shows that we are using a mix of Python out-of-the-box 🎁 libraries (os, shutil, gzip) alongside some custom ones (gdown, numpy). Gdown allows us to download files from Google Drive, which is where we saved our modified dataset. Gzip helps us unzip downloaded files. Shutil copies the downloaded files in the right location, and we'll use os to delete unneeded files.

import os
import shutil

import gdown
from numpy import genfromtxt

# Download file from Google Drive
# This file is based on data from: http://insideairbnb.com/get-the-data/
file_id_1 = "13fyESiH1ZEnMV6eabAyhe20t4W6peEWK"
downloaded_file_1 = "WK1_Airbnb_Amsterdam_listings_proj.csv"

# Download the file from Google Drive
gdown.download(id=file_id_1, output=downloaded_file_1)

## Preprocessing the Dataset
Getting this particular dataset loaded is a tad bit different from what we learned in this week's content. This time we only have 1 CSV file, so we do not need to merge it. So because there is no need to merge we can move right into data preprocessing!

#### Task 1: Find your delimiter

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/loading-inspect-dataset#corise_cl9jxhls800403b6pf9k9nh9e)

Inspect the csv file we just downloaded and look at the type of delimiter it has. Once you've found the right delimiter, use the dtype "unicode". 

from numpy import genfromtxt

my_data = genfromtxt(downloaded_file_1, delimiter="|", dtype="unicode")

Next, output the first four columns for inspection to see if you've got the data formatted how you'd like.

print(my_data[:, :4])

Awesome! But notice our data is aligned a little differently than how we saw in the course materials. It's like we shifted our dataset by 90 degrees! You'll have to fix this a little bit later.

#### Task 2: Clean it up

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/merging-datasets#corise_cl9k2znzw000t3b6poq4jg6o5)

In order for your calculations to run correctly, you need to have only the "relevant" numbers/entries present in your dataset. This means no headers, footers, redudant IDs, etc. Can you remove the first row and column, since you won't be needing them? Verify your work by again by printing out the first four columns.

# Remove the first column and row
matrix = my_data[1:, 1:]

# Print out the first four columns
print(matrix[:, :4])

print(my_data.shape)

print('Datatype:', my_data.dtype)

print(matrix.shape)

#### Task 3: Wide to long

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/loading-inspect-dataset#corise_cl9jzn1sx007e3b6p564y4yec)

As stated previously, our dataset is shifted by 90 degrees. Let's shift it another 90 degrees to get it back to how we'd expect, which is in a much more readable format. Please find in the course material in CoRise the correct operation that you'd need to use to do that. Again, verify your work by printing out of the first five rows.

# Shift the matrix by 90 degrees
matrix = matrix.transpose()
# Print out the first five rows
# Entries: airbnb_id, price_usd, latitude, longitude
print(matrix[:5,:])
matrix.shape

Now that all data is loaded properly, let's clean it up a bit like we did before by removing string characters and setting the right type 😁.

#### Task 4: That character is not appropriate

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/merging-datasets#corise_cl9k3ccc7001v3b6pz8a29ekr)

String characters like commas and dollar signs are yet again present in the dataset. Please find in the CoRise course materials the correct operation to filter out these two string characters from the dataset.

# Remove the dollar sign
matrix = np.char.replace(matrix, "$", "")

# Remove the comma
matrix = np.char.replace(matrix, ",", "")

matrix[:5, :]

Awesome! Now the dataset contains only numerical values allowing us to perform numerical operations... at least if we set the type right!

#### Task 5: Verification is the key to success!

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/merging-datasets#corise_cl9k3ccc7001v3b6pz8a29ekr)

Let's verify our matrix to confirm there are no more string characters present. Check to see if a dollar sign or comma still appears anywhere in the matrix. 


# Check if the dollar sign is in our dataset
matrix[(np.char.find(matrix, "$") > -1)]

# Check if the comma sign is in our dataset
matrix[(np.char.find(matrix, ",") > -1)]

#### Task 6: Are you my type?

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/merging-datasets#corise_cl9k3ccc7001v3b6pz8a29ekr)

Enabling numerical operations (calculations) requires you to change the `dtype` from string/Unicode characters to [float of 32-bit precision](https://numpy.org/doc/stable/user/basics.types.html?highlight=data%20types). Please change the dtype of the matrix to float32.


# Change Unicode to float32
matrix = matrix.astype(np.float32)

# Print out the first five rows (and inspect the dtype for correctness)
# Entries: airbnb_id, price_usd, latitude, longitude
print(matrix[:5,:])
matrix.shape

## The Price Is Right

<center>
  <img src=https://wwwimage-tve.cbsstatic.com/thumbnails/photos/w400-q80/blog/tpir-logo-promo_0_0.jpg width="500" align="center" />
</center>
<br/>

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/math-superpowers#corise_cl9k5z30h005w3b6p5jv0vs82)

Since all our values in the matrix are now recognized as numbers, we can perform some awesome calcultions! 

Our next objective is to change the currency from US dollars to another currency. This can be any currency you like, except for the US dollar. Let's first import the library that helps us to make these conversions. Then let's have another look at the first 5 rows of our matrix.

from currency_converter import CurrencyConverter

cc = CurrencyConverter()

# Entries: airbnb_id, price_usd, latitude, longitude

print(matrix[:5,:])
print(matrix[:, 1])
matrix.shape


The currency conversion calculations you'll be performing should be applied to the second column. 

As a reminder, you should use the number "1" which represents the second column of the matrix, since indexes start at the number zero.

Please only output the second column below:

print(matrix[:, 1])

#### Task 7: Pick any currency

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/math-superpowers#corise_cl9k5z30h005w3b6p5jv0vs82)

The tool you'll be using has a total of 42 currencies. Please select one of them, and use it to convert the dollars into your chosen currency. You can check which are available by running:

cc.currencies

A suggestion for those who don't know which to choose: Feel free to use GBP.

# Get the rate of conversaton from the US dollar to your currency of choice



gbp_rate = cc.convert(1, "USD", "GBP")

# Multiply the dollar column by your currency of choice
matrix[:, 1] = matrix[:, 1] * gbp_rate


print(matrix[:, 1])

#### Task 8: Inflation!

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/math-superpowers#corise_cl9k5z30h005w3b6p5jv0vs82)

Recent inflation all around the world has caused many companies to raise their prices. Consequently, Airbnb listings have also raised their prices by a certain amount. Find the 2022 annual inflation rate for your currency of choice. If you can't find the inflation rate online, use 7% as value. Apply this inflation rate to our newly updated prices.

# Multiply the dollar column by the inflation percentage (1.00 + inflation)
matrix[:, 1] = matrix[:, 1] * (1.00 + 0.07)


matrix[:, 1]

#### Task 9: Too many decimals!

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/math-superpowers#corise_cl9k5z30h005w3b6p5jv0vs82)

You might have some prices longer than two decimals after changing the currency and adjusting the price for inflation. Please round the prices **down** to the nearest two decimals using a NumPy native function. [Here's a hint.](https://numpy.org/doc/stable/reference/generated/numpy.round_.html)

# Round down the new currency column to 2 decimals
matrix[:, 1] = np.round(matrix[:, 1], 2)


print(matrix[:, 1])

## Where u (want to be) at?


<center>
  <img src=https://media.timeout.com/images/105504583/750/422/image.jpg width="500" align="center" />
</center>
<br/>

Amsterdam is a city with a long history and a rich international culture, so there is always [something interesting to see and do](https://www.iamsterdam.com/en/see-and-do/things-to-do/top-20-things-to-do-in-amsterdam). What if you were to visit Amsterdam? You'd probably want to have your Airbnb close to your favorite spot!

#### Task 10: Choose your location

Look up a place you'd like to visit in Amsterdam's city center, along with its longitude and latitude. We want to save this for choosing an Airbnb listing to our liking. You can get coordinates from [Google](https://www.google.com/) by searching like so:

<center>
  <img src=https://i.ibb.co/XXdkH3z/Screen-Shot-2022-10-24-at-2-42-54-PM.png width="500" align="center" />
</center>
<br/>




# Favorite location
latitude = 52.3600
longitude = 4.8852

## Listing All Listings

<center>
  <img src=https://images0.persgroep.net/rcs/vnd5KBhggcKV72YJjpLWH_-xljU/diocontent/131036963/_crop/34/170/1378/778/_fitwidth/763?appId=93a17a8fd81db0de025c8abd1cca1279&quality=0.8&desiredformat=webp width="500" align="center" />
</center>
<br/>

Imagine Airbnb Amsterdam decided to deviate from Airbnb Global and provide a feature on their website that showed the best listings for you based on the locations you were planning to visit. Wouldn't it make sense to choose a place to stay in a location closest to where you're likely to go most often?

So this is the most exciting part: You're going to calculate just that! You will limit your results to your favorite location in Amsterdam (as chosen above) and the surrounding available Airbnb listings using math and NumPy!

We've already provided you with the math calculations down below. Please make sure to run that method!

You'll have to use this method in a `for` loop or by using [`np.vectorize`](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html) as was shown on CoRise. 

import math

def from_location_to_airbnb_listing_in_meters(lat1: float, lon1: float, lat2: list, lon2: list):
    # Source: https://community.esri.com/t5/coordinate-reference-systems-blog
    # /distance-on-a-sphere-the-haversine-formula/ba-p/902128
    
    R = 6371000  # Radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # Output distance in meters

    return round(meters, 0)

#### Task 11: Loop or vectorize!

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/math-superpowers#corise_cl9k5z30h005w3b6p5jv0vs82)

Please implement a `for` loop (or vectorize) the `from_location_to_airbnb_listing_in_meters` function. ***For now just calculate these numbers. Don't add it as a new column to your matrix.***

# Create a loop or vectorized way to calculate the distance,
# going over all latitude and longitude entries in the dataset
conv_to_meters = np.vectorize(from_location_to_airbnb_listing_in_meters)

conv_to_meters(latitude, longitude, matrix[:, 2], matrix[:, 3])

Now let's use the `timeit` function to see how quickly the code is ran!

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/math-superpowers#corise_cl9k5z30h005w3b6p5jv0vs82)

%%timeit -r 4 -n 100

# Allow a Python function to be used in a (semi-)vectorized way
conv_to_meters = np.vectorize(from_location_to_airbnb_listing_in_meters)

# Apply the function, use timeit
conv_to_meters(latitude, longitude, matrix[:, 2], matrix[:, 3])

## Can We Do It Faster?

<center>
  <img src=https://upload.wikimedia.org/wikipedia/commons/9/9f/Serengeti_Lion_Running_saturated.jpg width="500" align="center" />
</center>
<br/>

Now you might be thinking to yourself, *can we do this faster*? 

The answer is ***YES***! 

But please remember that optimization is always a trade-off between the need for speed and the need for delivery of your results!

---

#### (Extra Credit)  Task 12: NumPy all the way!

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/math-superpowers#corise_cl9k5z30h005w3b6p5jv0vs82)

Now convert the `from_location_to_airbnb_listing_in_meters` function into a pure NumPy function. You can do this by changing all the imported math functions into their NumPy variant. ***For now just calculate these numbers. Don't add it as a new column to your matrix.***

def from_location_to_airbnb_listing_in_meters(lat1: float, lon1: float, lat2: list, lon2: list):   
    R = 6371000  # Radius of Earth in meters
    phi_1 = np.radians(lat1) # CHANGE THIS
    phi_2 = np.radians(lat2) # CHANGE THIS

    delta_phi = np.radians(lat2 - lat1) # CHANGE THIS
    delta_lambda = np.radians(lon2 - lon1) # CHANGE THIS

    a = (
        np.sin(delta_phi / 2.0) ** 2 # CHANGE THIS
        + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2 # CHANGE THIS (3x)
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) # CHANGE THIS (3x)

    meters = R * c  # Output distance in meters

    return np.round(meters, decimals = 0) # CHANGE THIS

# Run the converted NumPy method and check if it works

# Allow a Python function to be used in a (semi-)vectorized way
from_location_to_airbnb_listing_in_meters(latitude, longitude, matrix[:, 2], matrix[:, 3]
)

#### (Extra Credit) Task 13: How much faster is it?

[*\[Related section on CoRise\]*](https://corise.com/course/python-for-data-science/v2/module/math-superpowers#corise_cl9k5z30h005w3b6p5jv0vs82)

Use the `timeit` function so we can compare it to the outcome of the prior task. This should be very similar to the `timeit`-code that was used below Task 11.


# Copy the code from Task 12 and add a timeit function above this comment
%%timeit -r 4 -n 100
# Allow a Python function to be used in a (semi-)vectorized way
from_location_to_airbnb_listing_in_meters(latitude, longitude, matrix[:, 2], matrix[:, 3]
 )

WOW! You see a massive speed-up just by switching your functions from default Python functions to their NumPy variants! Awesome!

---

## Prep the Dataset for Download!


Now that we've created a function to calculate the distance in meters for every Airbnb listing, we'll perform this calculation on the entire dataset and add the outputs to the matrix as a new column.

Next to that, we'll add another column that contains only ones and zeros to represent the "color" of an entry/row. This column can be used later if you want to turn this dataset into an app using [Streamlit](https://streamlit.io/). This resource is great for when you want to translate your Python projects into an interactive website. More on that in the next section.

As you'll see from the code, we'll also add our favorite location as an entry/row. (We've selected the coordinates of the Rijksmuseum. Feel free to change it to your favorite location in Amsterdam). 




# Run the previous method
meters = from_location_to_airbnb_listing_in_meters(
    latitude, longitude, matrix[:, 2], matrix[:, 3]
)

# Add an axis to make concatenation possible
meters = meters.reshape(-1, 1)

# Append the distance in meters to the matrix
matrix = np.concatenate((matrix, meters), axis=1)

# Append a color to the matrix
colors = np.zeros(meters.shape)
matrix = np.concatenate((matrix, colors), axis=1)

# Append our entry to the matrix
fav_entry = np.array([1, 0, 52.3600, 4.8852, 0, 1]).reshape(1, -1) # Change coordinates to your favorite location
matrix = np.concatenate((fav_entry, matrix), axis=0)

# Entries: airbnb_id, price, latitude, longitude,
# meters from favorite point, color
matrix[:5, :]

# Export the data to use in the primer for next week
np.savetxt("WK1_Airbnb_Amsterdam_listings_proj_solution.csv", matrix, delimiter=",")

Great! By running all the cells above, you've saved the matrix here on your Google Colab instance. Let's now look into how to download the dataset to your local machine.

### Download the Dataset to Your Local Machine!

Google Colab comes with its own Python packages, allowing us to quickly download generated files like so:

from google.colab import files

# Download the file locally
files.download('WK1_Airbnb_Amsterdam_listings_proj_solution.csv')
