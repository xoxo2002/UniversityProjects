var fname = document.getElementById("fname");
var lname = document.getElementById("lname");
var suggestion = document.getElementById("suggestion");

fname.addEventListener("input", function(){checkeach(fname);});
lname.addEventListener("input", function(){checkeach(lname);});

//error messages 
const REQUIRED = "*This field cannot be blank";
const ALPHABETS = "*Alphabets only";

function checkeach(input) {
    if(input.checkValidity()){
        input.style.borderColor = "green";
        const small = input.parentNode.lastElementChild;
        small.style.color = ("green");
        small.innerHTML = "*valid";
    }
    else if(!input.checkValidity()){
        input.setCustomValidity("");
        console.log(input.checkValidity());
        error(input);
        }
    }
    
function error(input){
    input.style.borderColor = "red";
    const small = input.parentNode.lastElementChild;
    small.style.color = ("red");
    //see if empty
    if(input.value==""){
        small.innerHTML = REQUIRED;
    }   
    else if(input ==fname){
        small.innerHTML = ALPHABETS;
    }
    else if(input ==lname){
        small.innerHTML = ALPHABETS;
    }           
}
