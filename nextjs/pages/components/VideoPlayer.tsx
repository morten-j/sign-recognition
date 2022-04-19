import ReactPlayer from "react-player";
import React from 'react'

type Props = {url: string}

const VideoPlayer = ({url}: Props) => {
  return (
    <ReactPlayer url={url} controls={true} />
  )
}

export default VideoPlayer