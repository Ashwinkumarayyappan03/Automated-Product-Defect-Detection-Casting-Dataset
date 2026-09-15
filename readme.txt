# **Adding Images to a `README.md` File on GitHub**

There are **three simple ways** to add images to your GitHub `README.md` file.

---

## **1. Standard Markdown Syntax**

Use the following Markdown syntax:

```markdown
![Image Alt Text](URL_or_Path_to_Image)
```

### **Example — Image Stored in Your Repository**

```markdown
![Project Logo](./images/logo.png)
```

### **Example — Image Hosted Externally**

```markdown
![Project Logo](https://example.com/logo.png)
```

---

## **2. HTML Syntax — Resizing or Centering Images**

HTML gives you more control over the **image size, alignment, and appearance**.

```html
<img src="./images/logo.png" alt="Project Logo" width="300"/>
```

You can change the image width by modifying the `width` value:

```html
<img src="./images/logo.png" alt="Project Logo" width="500"/>
```

---

## **3. Easiest Way on GitHub — Drag & Drop**

The easiest method is to upload the image directly while editing your `README.md`.

### **Steps**

**1.** Open your **GitHub repository**.

**2.** Open the `README.md` file and click **Edit**.

**3.** Drag and drop your image directly into the GitHub editor.

**4.** GitHub will automatically **upload the image** and generate the required Markdown code.

**5.** Commit the changes.

---

### **Recommended**

For simple images, use:

```markdown
![Image Description](./images/image.png)
```

For **resizing or alignment**, use HTML:

```html
<img src="./images/image.png" alt="Image Description" width="500"/>
```
