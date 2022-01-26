# gdiv
Workflow to quantify experimental and simulated angular divergence of solar simulators

## Getting started

### On your Linux machine
1. Open your favorite shell, for example, good old
   [Bourne Again SHell, aka, `bash`](https://www.gnu.org/software/bash/),
   the somewhat newer
   [Z shell, aka, `zsh`](https://www.zsh.org/),
   or shiny new
   [`fish`](https://fishshell.com/).
2. Install [Git](https://git-scm.com/) by running
   `sudo apt install git-all` on [Debian](https://www.debian.org/)-based
   distributions like [Ubuntu](https://ubuntu.com/), or
   `sudo dnf install git` on [Fedora](https://getfedora.org/) and closely-related
   [RPM-Package-Manager](https://rpm.org/)-based distributions like
   [CentOS](https://www.centos.org/). For further information see
   [Installing Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
3. Clone the source code by running
   `git clone https://github.com/bbuenoun/gdiv.git` and navigate
   into the new directory `gdiv` by running `cd gdiv`.

#### With Docker and GNU Make (the easy way)
4. Install [Docker Desktop](https://www.docker.com/products/docker-desktop), and
   [GNU Make](https://www.gnu.org/software/make/).
5. List all GNU Make targets by running `make help`.
6. Drop into `bash` with the working directory `/app`, which
   is mounted to the host's working directory, inside a fresh Docker container
   based on Debian Linux with radiance installed by running `make shell`.
   If necessary, the Docker image is (re)built automatically, which takes
   a while the first time.
7. Execute gdiv by running `python gdiv.py`.
8. Drop out of the container by running `exit` or pressing `Ctrl-D`.

#### Without Docker (the hard way)
4. Install [Radiance](https://www.radiance-online.org/) version 5.2 and
   [Python](https://www.python.org/) version 3.7.3 and
   [pip](https://pip.pypa.io/en/stable/) version 18.1.
   [graphviz](https://www.graphviz.org) version 2.40.1.
5. Install
   * the code formatter [Black](https://github.com/psf/black),
   * the static type checker [mypy](http://mypy-lang.org),
   * the testing automator [pytest](https://docs.pytest.org)
   * the linter [Pylint](https://www.pylint.org/), and
   * the dead code finder [Vulture](https://github.com/jendrikseipp/vulture).
6. Install all packages in `requirements.txt` with
   [pip](https://pip.pypa.io/en/stable/) by running
   `pip install -r requirements.txt`.
7. Generate `Ztran.*.so` by running `f2py -c Ztran.pyf Ztran.f90`.
8. Execute gdiv by running `python gdiv.py`.

## Making code changes
In a nutshell, you do the following:
1. `git checkout develop`: Check-out, that is, switch to, the branch `develop`.
1. `git pull -p`: Pull up-stream changes, that is, sync your local copy with
   the remote repository.
1. `git checkout -b feature-x`: Create a new feature branch
1. Make changes to the code and the tests (use [type
   hints](https://docs.python.org/3/library/typing.html) for both) and
   regularly
   * format the code with
     [Black](https://github.com/psf/black) by running `make format` or
     equivalently `black --target-version py37 .`,
   * check for code smells with
     [Pylint](https://www.pylint.org/) by running `make lint` or equivalently
     `pylint *.py`,
   * find dead code with
     [Vulture](https://github.com/jendrikseipp/vulture) by running `make dead` or
     equivalently `vulture .`,
   * do static type checks with
     [mypy](http://mypy-lang.org/) by running `make types` or equivalently `mypy
     .`, and
   * test the code with [pytest](https://docs.pytest.org/en/) by running
     `make tests` or equivalently `python -m pytest tests` and `python -m pytest
     --assert=plain --doctest-continue-on-failure --doctest-modules *.py`.
   * document your code with
     [Google Style Python Docstrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
     (see also
     [this](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
     and
     [that](https://www.python.org/dev/peps/pep-0257/)),
     add
     [doctest](https://docs.python.org/3/library/doctest.html)
     usage examples, and generate HTML documentation by running `make docs` or
     equivalently `sphinx-apidoc -f -o docs/source . && sphinx-build -b html
     docs docs/html`
   and once in a while run slow but more elaborate tests with `make slowtests`
   or equivalently `python -m pytest --runslow tests`.
1. `git add .`: Stage your changes.
1. `git status`: Review the staged files (which files are staged as having been
   changed, added, or deleted).
1. `git diff --cached`: Review the staged line changes.
1. `git commit -m '...'`: Commit the staged changes, where the given message
   describes the changes.
1. `git push -u origin feature-x` (or just `git push` on subsequent changes):
   Push the feature branch to the remote repository.
1. Open [gdiv's GitHub](https://github.com/bbuenoun/gdiv) in
   the browser and create a merge request for the pushed branch marking the
   source branch to be deleted after successful merge and possibly linking the
   request to related issues. If the merge request is still work in progress,
   prefix its title with 'WIP'.
1. Add a changelog entry for your merge request, do some further changes `git
    add .` them, `git commit -m '...'` them, and `git push` them.
1. Once the feature is finished, on
   [gdiv's GitHub](https://github.com/bbuenoun/gdiv)
   remove the prefix 'WIP' from the merge request's title, ask for it to be
   reviewed, and, once it has been approved, merge it.

   If merging is not possible due to merge conflicts, then run `git fetch
   origin develop:develop` to pull up-stream changes of the `develop` branch
   into your local copy of it and `git rebase develop` to re-base your feature
   branch onto the new version of `develop`, that is, replay any changes you
   made on the new version of `develop`, which results in a history as if you
   had created the feature branch based on the new `develop` in the first
   place.

   The re-basing process is interactive: It stops at conflicts that cannot be
   resolved automatically and asks you to do so. Use `git status` to list
   files with non-resolved conflicts. Open those files, have a look at
   conflicting lines which are separated by the conflict dividers `<<<<<<<
   HEAD`, `=======`, and `>>>>>>> feature-x`, merge those lines manually and
   remove the dividers. Stage the manually merged files with `git add .` and
   continue the re-basing process with `git rebase --continue`. When the
   process is finished, run `make -k format lint dead types tests` to make
   sure that everything is fine and if it is force push the changes with `git
   push -f` which rewrites the upstream history with your shiny new local one
   that does not have any conflicts with the `develop` branch. Finally, merge
   the merge request. If for some reason you want to abort the re-basing
   process, then run `git rebase --abort`. For further details see
   [Git merge conflicts](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts).
1. `git checkout develop && git pull -p && git branch -d feature-x`: Check-out
   the branch `develop`, pull up-stream changes, and remove the local feature
   branch.
1. `make tests`: Run all tests. If some fail, which is possible due to changes
   that were merged into `develop` after the branch `feature-x` was created,
   then fix the code in a new branch `fixes-x` with the same workflow as above.

This way of working with `git` is known as the [Gitflow
workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
which you should internalize. You should also
* read excerpts of [Get started with GitHub](https://docs.github.com/en/github/getting-started-with-github/),
* familiarize yourself with [Semantic Versioning](https://semver.org/),
* know how to [Keep a changelog](https://keepachangelog.com/en/1.0.0/),
* learn [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/),
* learn how to [Use branches](https://www.atlassian.com/git/tutorials/using-branches),
* learn the differences between [Merging and Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing),
* learn [Making a Pull Request](https://www.atlassian.com/git/tutorials/making-a-pull-request), and
* adhere to the [PEP8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).

The essence of the Gitflow workflow: There are two permanent branches `master`
and `develop`, where `master` is for releases and `develop` for active
development towards the next release. Features are developed in their own
branches based on `develop` and merged into `develop` when they are finished.
Hot-fixes are developed in their own branches based on `master` and merged into
`master` when they are finished. New releases are made by merging `develop`
into `master`. Each hot-fix and normal release is tagged with a version.

Each feature and hot-fix branch has an accompanying merge request, which,
when feasible, is linked to one or multiple.
Before a merge request is accepted, it is reviewed to make sure that the
changes are properly implemented, well tested, and a changelog entry has
been added.

## How-to

### Use Git
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/), aka, [Gitflow Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
- [Making a Pull Request](https://www.atlassian.com/git/tutorials/making-a-pull-request)
- [Merging vs. Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
- [Using branches](https://www.atlassian.com/git/tutorials/using-branches)
- [Git merge conflicts](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts)
### Test code
- [What is unit testing and what are its benefits?](https://stackoverflow.com/questions/1383/what-is-unit-testing/1398#1398)
- [The Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)
### Use Docker
- [Docker for development, why, and how?](https://www.reddit.com/r/docker/comments/982cag/docker_for_development_why_and_how/)
- [Why and How to Use Docker for Development](https://medium.com/travis-on-docker/why-and-how-to-use-docker-for-development-a156c1de3b24)
- [Efficient development with Docker and docker-compose](https://hackernoon.com/efficient-development-with-docker-and-docker-compose-e354b4d24831)
- [A Practical Introduction to Docker Compose](https://hackernoon.com/practical-introduction-to-docker-compose-d34e79c4c2b6)
### Use GNU Make
- [Running Make](https://swcarpentry.github.io/make-novice/reference.html)
- [The Lost Art of the Makefile](https://www.olioapps.com/blog/the-lost-art-of-the-makefile/)
### Version releases
- [Semantic Versioning](https://semver.org/)
- [Keep a changelog](https://keepachangelog.com/en/1.0.0/)
### Do what?
- [Make a README](https://www.makeareadme.com/)
- [Get started with GitHub](https://docs.github.com/en/github/getting-started-with-github/)
- View HDR files: Use the image viewer `ximage` that comes with radiance.
