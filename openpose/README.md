# Openpose docker builder

The Dockerfile builds openpose along with <https://github.com/martinloenne/sign-language-recognition-service>.

However some changes to sign-language-recognition-service had to be made since they use roundabout ways to import openpose along with having windows-style paths in the code.

Therefore, the folder `overrides/` has been made which is copied into sign-language-recognition-service which makes the files get put into their respective paths, overriding their counterparts so that the service can run in a linux environment.  
