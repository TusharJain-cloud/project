const express = require('express');
const fetch = require('node-fetch');

const app = express();

app.set('view engine', 'ejs');
app.use(express.static('public'));
app.use(express.urlencoded());

app.get("/", (req, res) => {
    res.render("home");
});

app.post("/movies", (req,res) => {

    const query = req.body.MovieName;
    const apiKey = '0a3fbecafd4b444c79b596344547dd83';
    const url = "https://api.themoviedb.org/3/search/movie?api_key=" + apiKey + "&query=" + query;

    fetch(url)
  .then((response) => {
    return response.json();
  })
  .then((myJson) => {
    console.log(myJson);
  });

});

app.listen(3000, ()=> {
    console.log("Server has started on port successfully");
  });
  

  