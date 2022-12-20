var x = 0;

function onAndOff(){
    if(x==0){
        document.getElementById('lightbulb').src='./pictures/orangelightbulb.svg';
        document.getElementById('OnOffButton').innerHTML="Press To Off";
        x = 1;
    }
    else{
        document.getElementById('lightbulb').src='./pictures/lightbulb.svg';
        document.getElementById('OnOffButton').innerHTML="Press Me";
        x = 0;
    }
}