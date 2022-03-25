import Image from 'next/image'
import betaPic from '../public/flunky strong.jpg'

export default function beta() {
    return (
        <>
            <h1>Kekekekek</h1>
            <Image
                src={betaPic}
                alt="Picture of the author"
                // width={500} automatically provided
                // height={500} automatically provided
                // blurDataURL="data:..." automatically provided
                // placeholder="blur" // Optional blur-up while loading
            />
            <p>Welcome to my homepage!</p>
        </>
    )
}