$(function() {
  const source = document.getElementById('autoComplete');
  const inputHandler = function(e) {
    if (e.target.value == "") {
      $('.movie-button').attr('disabled', true);
    } else {
      $('.movie-button').attr('disabled', false);
    }
  }
  if (source) {
    source.addEventListener('input', inputHandler);
  }

  $('.movie-button').on('click', function() {
    var title = $('.movie').val();
    if (title == "") {
      $('.results').css('display', 'none');
      $('.fail').css('display', 'block').text('Please enter a movie title!');
    } else {
      load_recommendations(title);
    }
  });
});

function setLoading(isLoading) {
  var loader = $("#loader");
  if (isLoading) {
    loader.css("display", "flex").attr("aria-hidden", "false");
  } else {
    loader.fadeOut(200, function() {
      loader.css("display", "none").attr("aria-hidden", "true");
    });
  }
}

function load_recommendations(title) {
  $('.results').css('display', 'none');
  $('.fail').css('display', 'none');
  setLoading(true);

  $.ajax({
    type: 'POST',
    url: '/get_all_movie_data',
    data: { 'name': title },
    dataType: 'json',
    success: function(response) {
      setLoading(false);
      if (response.status === 'success') {
        $('.results').html(response.html).fadeIn(400);
        $('#autoComplete').val('');
        $(window).scrollTop(0);
      } else {
        $('.fail').css('display', 'block').text(response.message);
      }
    },
    error: function(xhr) {
      setLoading(false);
      var err_msg = 'Sorry! The movie you requested is not in our database. Please check the spelling or try another movie.';
      if (xhr.responseJSON && xhr.responseJSON.message) {
        err_msg = xhr.responseJSON.message;
      }
      $('.fail').css('display', 'block').text(err_msg);
    }
  });
}

// --- NEW FUNCTION: Makes recommended movie cards clickable ---
function recommendcard(e) {
  // Grab the movie title from the 'title' attribute of the clicked card
  var title = e.getAttribute('title');
  
  if (title) {
    // Fill the search box with the new title
    $('#autoComplete').val(title);
    
    // Call the recommendation API directly with the new title
    load_recommendations(title);
  }
}
