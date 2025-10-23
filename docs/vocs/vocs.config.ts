import React from 'react'
import { defineConfig } from 'vocs'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import remarkMdxDisableExplicitJsx from 'remark-mdx-disable-explicit-jsx'

import { bookSidebar, specsSidebar } from './sidebar'

export default defineConfig({
  title: 'OpenVM',
  logoUrl: '/OpenVM-horizontal.svg',
  iconUrl: '/OpenVM-favicon.svg',
  ogImageUrl: '/OpenVM-horizontal.svg',
  sidebar: {
    '/book/': bookSidebar,
    '/specs/': specsSidebar
  },
  basePath: '/',
  topNav: [
    { text: 'Book', link: '/book/getting-started/introduction' },
    { text: 'Specs', link: '/specs/openvm/overview' },
    { text: 'Whitepaper', link: 'https://openvm.dev/whitepaper.pdf' },
    {
      text: 'Rustdocs',
      link: `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL || 'localhost:3000'}/docs/openvm`
    },
    { text: 'GitHub', link: 'https://github.com/openvm-org/openvm' },
    {
      text: 'v1.4.1',
      items: [
        {
          text: 'Releases',
          link: 'https://github.com/openvm-org/openvm/releases'
        },
      ]
    }
  ],
  socials: [
    {
      icon: 'github',
      link: 'https://github.com/openvm-org/openvm',
    },
    {
      icon: 'telegram',
      link: 'https://t.me/openvm',
    },
  ],
  sponsors: [
    {
      name: 'Collaborators',
      height: 120,
      items: [
        [
          {
            name: 'Axiom',
            link: 'https://axiom.xyz',
            image: '',
          },
        ]
      ]
    }
  ],
  markdown: {
    remarkPlugins: [[remarkMath, { singleDollarTextMath: true }]],
    rehypePlugins: [[rehypeKatex, {
        // Strict mode can help with parsing
        strict: false,
        // Trust all LaTeX commands
        trust: true
      }]],
  },
  vite: {  
    plugins: [  
      {  
        name: 'exclude-llms',  
        configResolved(config) {  
          // Cast to mutable to modify the readonly plugins array  
          const mutableConfig = config as any  
          mutableConfig.plugins = config.plugins.filter(  
            plugin => plugin && plugin.name !== 'llms'  
          )  
        }  
      }  
    ],
  },
  theme: {
    accentColor: {
      light: '#1f1f1f',
      dark: '#ffffff',
    }
  },
  editLink: {
    pattern: "https://github.com/openvm-org/openvm/edit/main/docs/vocs/docs/pages/:path",
  }
})
