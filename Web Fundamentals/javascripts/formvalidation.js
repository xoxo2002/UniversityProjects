var fname = document.getElementById("fname");
var lname = document.getElementById("lname");
var age = document.getElementById("age");
var email = document.getElementById("email");

fname.addEventListener("input", function(){checkeach(fname);});
lname.addEventListener("input", function(){checkeach(lname);});
age.addEventListener("input", function(){checkeach(age);});
email.addEventListener("input", function(){checkeach(email);});

                //error messages 
const REQUIRED = "*This field cannot be blank";
const ALPHABETS = "*Alphabets only";
const BELOW_AGE = "*Must be at least 10 years old to register";
const INVALID_EMAIL = "*Must be in format username@domain.com";

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
    else if(input == age){
        small.innerHTML = BELOW_AGE;
    }
    else if(input == email){
        small.innerHTML = INVALID_EMAIL;
    }                
}
