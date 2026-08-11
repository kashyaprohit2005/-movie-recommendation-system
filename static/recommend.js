$(function() {
  const source = document.getElementById('autoComplete');
  const inputHandler = function(e) {
    if(e.target.value==""){
      $('.movie-button').attr('disabled', true);
    }
    else{
      $('.movie-button').attr('disabled', false);
    }
  }
  if (source) {
    source.addEventListener('input', inputHandler);
  }

  $('.movie-button').on('click', function(){
    var title = $('.movie').val();
    if (title=="") {
      $('.results').css('display','none');
      $('.fail').css('display','block');
      $('.fail').text('Please enter a movie title!');
    }
    else {
      load_details(title);
    }
  });
});

function load_details(title){
  $('.results').css('display','none');
  $('.fail').css('display','none');
  $("#loader").fadeIn();
  
  $.ajax({
    type: 'GET',
    url: '/api/search_movie?title=' + encodeURIComponent(title),
    success: function(movie){
      if(movie.results.length<1){
        $('.fail').css('display','block');
        $('.fail').text('Sorry! The movie you requested is not in our database. Please check the spelling or try with other movies!');
        $("#loader").delay(500).fadeOut();
      }
      else{
        $("#loader").fadeIn();
        $('.fail').css('display','none');
        $('.results').delay(1000).css('display','block');
        var movie_id = movie.results[0].id;
        var movie_title = movie.results[0].original_title;
        movie_recs(movie_title, movie_id);
      }
    },
    error: function(){
      alert('Error searching movie details through proxy');
      $("#loader").delay(500).fadeOut();
    }
  });
}

function movie_recs(movie_title, movie_id){
  $.ajax({
    type:'POST',
    url:"/similarity",
    data:{'name': movie_title},
    success: function(recs){
      if(recs=="Sorry! The movie you requested is not in our database. Please check the spelling or try with other movies."){
        $('.fail').css('display','block');
        $('.fail').text(recs);
        $("#loader").delay(500).fadeOut();
      }
      else {
        $('.fail').css('display','none');
        $('.results').delay(1000).css('display','block');
        var movie_arr = recs.split('---');
        var arr = [];
        for(const move in movie_arr){
          arr.push(movie_arr[move]);
        }
        get_movie_details(movie_id, arr, movie_title);
      }
    },
    error: function(){
      alert("Error getting movie similarity");
      $("#loader").delay(500).fadeOut();
    }
  });
}

function get_movie_details(movie_id, arr, movie_title) {
  $.ajax({
    type:'GET',
    url:'/api/movie_details/' + movie_id,
    success: function(movie_details){
      show_details(movie_details, arr, movie_title, movie_id);
    },
    error: function(){
      alert("API Error in movie details");
      $("#loader").delay(500).fadeOut();
    }
  });
}

function show_details(movie_details, arr, movie_title, movie_id){
  var imdb_id = movie_details.imdb_id;
  var poster = 'https://image.tmdb.org/t/p/original'+movie_details.poster_path;
  var overview = movie_details.overview;
  var genres = movie_details.genres;
  var rating = movie_details.vote_average;
  var vote_count = movie_details.vote_count;
  var release_date = new Date(movie_details.release_date).toDateString().split(' ').slice(1).join(' ');
  var runtime = parseInt(movie_details.runtime);
  var status = movie_details.status;
  var genre_list = [];
  for (var genre in genres){
    genre_list.push(genres[genre].name);
  }
  var my_genre = genre_list.join(", ");
  if(runtime%60==0){
    runtime = Math.floor(runtime/60)+" hour(s)"
  }
  else {
    runtime = Math.floor(runtime/60)+" hour(s) "+(runtime%60)+" min(s)"
  }

  // Get cast details
  var top_casts = [];
  var top_cast_ids = [];
  var cast_names = [];
  var cast_chars = [];
  var cast_profiles = [];

  $.ajax({
    type:'GET',
    url:'/api/movie_credits/' + movie_id,
    success: function(my_movie_cast){
      if(my_movie_cast.cast.length >= 8){
        top_casts = my_movie_cast.cast.slice(0, 8);
      } else {
        top_casts = my_movie_cast.cast;
      }
      for(var i=0; i<top_casts.length; i++){
        top_cast_ids.push(top_casts[i].id);
        cast_names.push(top_casts[i].name);
        cast_chars.push(top_casts[i].character);
        cast_profiles.push(top_casts[i].profile_path ? "https://image.tmdb.org/t/p/original" + top_casts[i].profile_path : "/static/default.jpg");
      }

      // Fetch cast bio details
      var cast_bdys = [];
      var cast_bios = [];
      var cast_places = [];
      var completed_requests = 0;

      if(top_cast_ids.length === 0){
        get_posters_and_render();
      }

      top_cast_ids.forEach(function(cast_id, idx){
        $.ajax({
          type:'GET',
          url:'/api/person/' + cast_id,
          success: function(person){
            cast_bdys[idx] = new Date(person.birthday).toDateString().split(' ').slice(1).join(' ');
            cast_bios[idx] = person.biography;
            cast_places[idx] = person.place_of_birth;
            completed_requests++;
            if(completed_requests === top_cast_ids.length){
              get_posters_and_render();
            }
          },
          error: function(){
            cast_bdys[idx] = "N/A";
            cast_bios[idx] = "N/A";
            cast_places[idx] = "N/A";
            completed_requests++;
            if(completed_requests === top_cast_ids.length){
              get_posters_and_render();
            }
          }
        });
      });

      function get_posters_and_render(){
        var movie_posters = [];
        var poster_requests = 0;

        arr.forEach(function(m_title, idx){
          $.ajax({
            type:'GET',
            url:'/api/search_movie?title=' + encodeURIComponent(m_title),
            success: function(m_data){
              if(m_data.results && m_data.results.length > 0 && m_data.results[0].poster_path){
                movie_posters[idx] = 'https://image.tmdb.org/t/p/original' + m_data.results[0].poster_path;
              } else {
                movie_posters[idx] = '/static/movie_placeholder.jpeg';
              }
              poster_requests++;
              if(poster_requests === arr.length){
                send_recommend_request();
              }
            },
            error: function(){
              movie_posters[idx] = '/static/movie_placeholder.jpeg';
              poster_requests++;
              if(poster_requests === arr.length){
                send_recommend_request();
              }
            }
          });
        });
      }

      function send_recommend_request(){
        var details = {
          'title': movie_title,
          'cast_ids': JSON.stringify(top_cast_ids),
          'cast_names': JSON.stringify(cast_names),
          'cast_chars': JSON.stringify(cast_chars),
          'cast_profiles': JSON.stringify(cast_profiles),
          'cast_bdys': JSON.stringify(cast_bdys),
          'cast_bios': JSON.stringify(cast_bios),
          'cast_places': JSON.stringify(cast_places),
          'imdb_id': imdb_id,
          'poster': poster,
          'genres': my_genre,
          'overview': overview,
          'rating': rating,
          'vote_count': vote_count.toLocaleString(),
          'release_date': release_date,
          'runtime': runtime,
          'status': status,
          'rec_movies': JSON.stringify(arr),
          'rec_posters': JSON.stringify(movie_posters)
        };

        $.ajax({
          type: 'POST',
          data: details,
          url: "/recommend",
          dataType: 'html',
          complete: function(){
            $("#loader").delay(500).fadeOut();
          },
          success: function(response) {
            $('.results').html(response);
            $('#autoComplete').val('');
            $(window).scrollTop(0);
          }
        });
      }
    }
  });
}
