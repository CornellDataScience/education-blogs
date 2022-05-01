This is a general purpose website serving as a host for blogs, tutorials, and other kinds of posts created by Cornell Data Science, particularly of the Education subteam. 

This was forked and adapted from [academicpages.github.io](https://github.com/academicpages.github.io), which is © 2017 Stuart Geiger and released under the MIT License. 
## To run locally (not on GitHub Pages, to serve on your own computer)

1. Clone the repository and made updates as detailed above
2. Make sure you have ruby-dev, bundler, and nodejs installed: `sudo apt install ruby-dev ruby-bundler nodejs`
3. Run `bundle clean` to clean up the directory (no need to run `--force`)
4. Run `bundle install` to install ruby dependencies. If you get errors, delete Gemfile.lock and try again.
5. Run `bundle exec jekyll serve` to generate the HTML and serve it from `localhost:4000`. Live updates do work, but you might have to be patient and give it time to refresh.

To deploy globally, simply commit and push!