## Step 3 — Extracting Faces and Saving them Locally

In the last step we saw how using openCV and HaaR cascade for face, we were able to detect faces and draw an bounding box around them in a input image. In this section we are going to build on top of the script created in the last section and add the code to extract those faces from the input image.

Before moving forward, lets investigate this _list of rectangles_ that got returned from  `detectMultiScale` method in last step. This list actually is a list of `pixel locations`, in the form of `Rect(x,y,w,h)`, for all the objects that were detected in the input image based on the cascade provided. We then used those `pixel locations` to create a bounding box around all the detected faces in the image.
<!-- TODO: I think that this explanation would work really well when explaining the code in the previous step to the reader. Can you move the `Rect(x,y,w,h)` explanation where it is first introduced? -->

#### Code

In this step we are simply going to build on the same logic and save each of the bounding box found in last step as a separate image. That's it. Lets see how to do it.

#### Code Walk through

As will be seen shortly, the script listed below is same as previous step for the most part except for a couple of lines. Lets look at those 

We only had to update the `for` loop we wrote earlier to as follows :

<!-- TODO: The above introduction to this step has some great details, and I think that you can make them shine through by doing some recasting to remove duplicate information. What do you think about using something like this?

"In the last step, you wrote code to use OpenCV and a HaaR cascade to detect and draw boxes around faces in an image. In this section, you will modify your code to extract the detected faces from the image into their own files. 

Open your `app.py` file with your text editor:

```command
nano app.py
```

Next, add the highlighted lines under the `cv2.rectangle` line:" -->

```python
...
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = image[y:y + h, x:x + w]
    print("[INFO] Object found. Saving to local !!")
    cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
...
```
<!-- TODO: Can you use variable highlighting to show the reader what will be add? Additionally, I'd like you to update the printed text to resolve some grammar and punctuation issues. Once updated here, please make sure to update the rest of the tutorial to use it. Can you update the printed text to something like this?

"[INFO] Object found. Saving locally.""
-->

That's it. As can be seen we only added two additional lines to achieve this. Let's go through them one by one.
<!-- TODO: Can you recast this to align more closely to our tone by removing the usage of "That's it."? Also, it looks like the reader should have added three lines of code. If that's correct, can you please update the section to reflect that? This will include adding an additional explanation for the `print` line.-->

```python
roi_color = image[y:y + h, x:x + w]
```

`roi_color` is the plot of `pixel locations` from list `faces` on the input color `image` for the first object.
<!-- TODO: Can you explain to the reader what the `[y:y + h, x:x + w]` bit does? -->

```python
cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
```

This code will save the plot as a new image using the `cv2.imwrite` method. It appends the width and height of the plot to the name of the image being written to keep the name unique in case there are multiple faces detected.

#### Putting it all together

Given here is the updated script. Please open the `app.py` we created earlier in your favorite IDE and replace its contents with code below.
<!-- TODO: Can you recast this to remove the H4? Once you implement my previous feedback about having the reader open the editor earlier on, I'd also like you to remove the mention of opening the text editor here. -->

```python
[label app.py]
import cv2
import sys

imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the image
faceCascade = cv2.CascadeClassifier(cascPath)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)

print("[INFO] Found {0} Faces.".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_color = image[y:y + h, x:x + w]
    print("[INFO] Object found. Saving to local !!")
    cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)

status = cv2.imwrite('faces_detected.jpg', image)
print("[INFO] Image faces_detected.jpg written to file-system:", status)
```
<!-- TODO: Can you please remove the comments from the code blocks? Please also be sure to tell the reader to save and close their text editor. -->

**The above code was sourced from OpenCV Documentation publicly available [here](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) except for few changes**

The idea behind the code added is quite simple. We already had the pixel locations for all the detected objects in a list `faces`. We simply had to plot them on the input image and save those plots. Will look at it in more depth in following section.
<!-- TODO: Can you recast this to remove the use of "simple"? That language can be demotivating to readers who did not find the code simple. Here's an example of how you could recast this:

To summarize, the added code uses the pixel locations for objects detected by the original code to extract the faces from the image into a new file. Now that you've updated the code, you are ready to run the script once more.-->

#### Running the Updated Script

Running the script is same as before. Make sure the image you want to process is in the same folder as the script `app.py`. Next invoke the script as follows:

```command
python app.py path/to/image
```
<!-- TODO: As a reminder, please be sure to use variable formatting on the image path. -->

If all has been good so far, we should expect to see following output at the time of execution.
<!-- TODO: Can you recast this to give the sentence a more confident tone? "You will see the similar output once your script is done processing the image:" -->

```command
[label Console Output]
$ python app.py beatles-spotlight-514890404.png
[INFO] Found 4 Faces !
[INFO] Object found. Saving to local !!
[INFO] Object found. Saving to local !!
[INFO] Object found. Saving to local !!
[INFO] Object found. Saving to local !!
[INFO] Image faces_detected.jpg written to file-system: True
```
<!-- TODO: Can you update this code block to only include output from the console? -->

Depending on how many faces were found in the input image, expect to see the line `[INFO] Object found. Saving to local !!` printed to the console that many times.
<!-- TODO: I think that you could shorten this sentence with some light recasting. "Depending on how many faces are in your sample image, you may see more or less output."-->

Looking at the contents of the working directory after the execution of the script, you should see head shots of all faces found in the input image.

![Directory Listing ](dir_listing.JPG)
<!-- TODO: As a reminder, please host any images in the tutorial on Imgur. -->

#### Output

You will now see head shots extracted from the input image collected in the working directory 

![Head shots](head_shots.JPG)

That's it.
<!-- TODO: Can you add a summary for the reader accomplished in this step? -->

## Step 3 — Other Applications and Further Reading.
<!-- TODO: Since there are no specific steps here, it would be best to tie this information in the conclusion. -->
As can be seen, with a few lines of code, we were able to setup an automated application which can detect, count and extract faces from an image. A interested user can extend the application for may real world usage. Some that come to mind:

* Using a `full_body_haar_cascade.xml` available at openCV's github repository, once can write a similar app to count the number of pedestrians in an input image. 
* Any automated cataloging system would benefit from an application of similar nature. Please note that the user would in that case need to train their own Haar Cascade model for that object.
* Automated polling system from images where the object to be detected is automatically detected using a application of similar nature. The list goes on.

