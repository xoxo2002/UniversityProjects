let slideIndex = 0;
showSlides(slideIndex);
//for the next or previous buttons
function nextSlides(n){
    showSlides(slideIndex += 1);
}
//for jumping to slides
function currentSlides(n){
    showSlides(slideIndex = n);
}
//logic to show which slide according to n
function showSlides(n){
    let i;
    let slides;
    slides = document.getElementsByClassName("slides");
    let dots = document.getElementsByClassName("dot");
    //for next slides
    if(n<0){
        slideIndex = slides.length - 1;
    }
    if(n >= slides.length){
        slideIndex = 0;
    }
    //undisplay other pictures and dots then only display the one u wan 
    for(i = 0; i < slides.length; i++){
        slides[i].style.display = "none";
        dots[i].className = dots[i].className.replace(" active", "");
    }
    slides[slideIndex].style.display = "block";
    dots[slideIndex].className += " active";
    }

    