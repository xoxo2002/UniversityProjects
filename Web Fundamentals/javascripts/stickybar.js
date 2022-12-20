const form = document.getElementById('subform');
            form.addEventListener('submit',(e)=>{
                e.preventDefault();
                var msg = document.getElementById('success');
                msg.style.display = "flex";
                setTimeout(function(){
                    msg.style.display = "none";}, 5000);
                    form.reset();
                    resetstyle();
            })

            function resetstyle(){
                var fields = document.getElementsByTagName("input");
                for(var i = 0; i < (fields.length)-2; i++){
                    const small = fields[i].parentNode.lastElementChild;
                    small.style.display = "none";
                    fields[i].style.borderColor = "gainsboro";
                }
            }

            //sticky success banner
            window.onscroll = function(){sticky()};

            var banner = document.getElementById("successcon");

            function sticky(){
                if(window.pageYOffset > 20){
                    banner.classList.add("sticky");
                }
                else{
                    banner.classList.remove("sticky");
                }
            }
