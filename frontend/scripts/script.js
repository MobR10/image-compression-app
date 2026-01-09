let imageInput = document.getElementById("imageInput");
let uploadedImg = document.getElementById("uploadedImg");
let processedImg = document.getElementById("processedImg");
let imgForm = document.getElementById("imgForm");
let factor = document.getElementById("factor");

let loadingEl = document.getElementById("loadingText");

imgForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    console.log("Submitting image");

    console.log(new FormData(imgForm).get("image"));

    loadingEl.textContent = "Loading...";
    const response = await fetch(imgForm.action, {
        method: "POST",
        body: new FormData(imgForm)
    });

    if (response.ok) {
        loadingEl.textContent = "Success!";
        const blob = await response.blob(); 
        processedImg.src = URL.createObjectURL(blob);
        console.log("Image received and displayed");
    } else {
        loadingEl.textContent = "Error"
        console.error("Upload failed");
    }
});

imageInput.addEventListener("change",(event) => {
    if(event?.target?.files && event.target.files[0]){
        uploadedImg.src = URL.createObjectURL(event.target.files[0]);
        processedImg.src = "";
        console.log(event.target.files);
    }
});

imageInput.addEventListener("click" , event =>{
    uploadedImg.src = "";
})


 

