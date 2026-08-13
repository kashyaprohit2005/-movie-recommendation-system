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

  // ==========================================
  // --- NEW TRAILER LOGIC ---
  // ==========================================
  
  // 1. Listen for clicks on the trailer button (works on injected HTML)
  $(document).on('click', '.watch-trailer-btn', async function(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const movieId = $(this).attr('data-movie-id');
    if (!movieId) return;

    const modal = $('#trailerModal');
    const iframe = $('#trailerIframe');
    const titleEl = $('#trailerTitle');
    const loader = $('#trailerLoader');
    const errorMsg = $('#trailerError');

    // Show modal and loading state
    iframe.hide().attr('src', '');
    errorMsg.hide();
    loader.show();
    titleEl.text('Fetching Trailer...');
    modal.addClass('active');

    try {
        const response = await fetch(`/api/trailer/${movieId}`);
        const data = await response.json();
        
        loader.hide();
        if (data.success) {
            titleEl.text(data.title);
            // Autoplay the YouTube video
            iframe.attr('src', `https://www.youtube.com/embed/${data.youtube_key}?autoplay=1`).show();
        } else {
            titleEl.text('Trailer Unavailable');
            errorMsg.text(data.message || 'We could not find a trailer for this movie.').show();
        }
    } catch (error) {
        loader.hide();
        titleEl.text('Connection Error');
        errorMsg.text('Failed to load trailer. Please try again later.').show();
    }
  });

  // 2. Close modal on X button
  $(document).on('click', '#closeTrailer', function() {
      $('#trailerModal').removeClass('active');
      $('#trailerIframe').attr('src', '');
  });

  // 3. Close modal on clicking outside the video
  $(document).on('click', '#trailerModal', function(e) {
      if (e.target === this) {
          $(this).removeClass('active');
          $('#trailerIframe').attr('src', '');
      }
  });

  // 4. Close modal on Escape key
  $(document).on('keydown', function(e) {
      if (e.key === 'Escape' && $('#trailerModal').hasClass('active')) {
          $('#trailerModal').removeClass('active');
          $('#trailerIframe').attr('src', '');
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

function recommendcard(e) {
  var title = e.getAttribute('title');
  if (title) {
    $('#autoComplete').val(title);
    load_recommendations(title);
  }
}
