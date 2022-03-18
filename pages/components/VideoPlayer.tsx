import ReactPlayer from "react-player";

export default function VideoPlayer() {
    return (
      <ReactPlayer url={'sign_videos/signvid.webm'} controls = {true} />
    );
}