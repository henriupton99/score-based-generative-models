# score-based-generative-models
In this work, you will study a recent, related, yet different approach to be able to synthesize new datapoints. This process does not rely on a low dimensional subset of vectors, but is rather aimed at generating new points in a certain space using noise sampled from the same space.

Deep Learning Course (Project Page) : https://marcocuturi.net/dl.html

Blog post and GitHub about the Process : 

https://yang-song.github.io/blog/2021/score/

https://github.com/yang-song/score_sde_pytorch


We work on MAESTRO music data that comes from TensorFlow : https://magenta.tensorflow.org/datasets/maestro#v300

Example of piano roll for one random MIDI File : 

![Example Image](./figures/piano_roll_760.png)

<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1500676579&color=%23ff9900&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/musichume" title="HUme" target="_blank" style="color: #cccccc; text-decoration: none;">HUme</a> · <a href="https://soundcloud.com/musichume/score-generative-model-music" title="Score Generative Model Doing Music" target="_blank" style="color: #cccccc; text-decoration: none;">Score Generative Model Doing Music</a></div>
