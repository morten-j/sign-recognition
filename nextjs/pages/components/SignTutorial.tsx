import React, { useEffect, useState } from "react"
import ReactPlayer from "react-player"
import useKeyPress from "./hooks/useKeyPress"

type Props = {
    signName: string,
    url: string,
    closeModal: () => void,
}

export default function SignTutorial({ signName, url, closeModal }: Props ) {

    useKeyPress("Escape", closeModal);

    return (
        <div onClick={closeModal} className="w-full h-full bg-gray-900 bg-opacity-80 top-0 fixed sticky-10s z-50 overflow-scroll">
                <div onClick={(e) => e.stopPropagation()} className="min-w-fit max-w-lg p-10 m-auto items-center mt-16 flex flex-col bg-white gap-5 rounded">

                    <button onClick={closeModal} className="text-gray-800 absolute top-2 right-2 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-800">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M18 6L6 18" stroke="currentColor" stroke-width="1.66667" stroke-linecap="round" stroke-linejoin="round" />
                            <path d="M6 6L18 18" stroke="currentColor" stroke-width="1.66667" stroke-linecap="round" stroke-linejoin="round" />
                        </svg>
                    </button>

                    <h1 className="text-xl font-semibold leading-7 text-center">
                        Make this sign: {signName}
                    </h1>
                    
                    <div className="flex max-h-96 max-w-fit">
                        <ReactPlayer url={url} playing={true} controls={false} loop={true} width="" height="" />
                    </div>
                    

                    <button onClick={closeModal} className="bg-blue-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
                        I'm Ready!
                    </button>

                </div>
        </div>
    )
}