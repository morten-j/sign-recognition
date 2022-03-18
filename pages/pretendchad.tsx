import type { NextPage } from 'next'
import Image from 'next/image'
import king from '../public/svelteking.png'

const Home: NextPage = () => {
  return (
    <>
      <h1>Kekekekek</h1>
      <Image
        src={king}
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

export default Home