import React from 'react'
import { defineConfig } from 'vocs'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

import { sidebar } from './sidebar'

export default defineConfig({
  title: 'OpenVM',
  logoUrl: '/OpenVM-horizontal.svg',
  iconUrl: '/OpenVM-favicon.svg',
  ogImageUrl: '/OpenVM-horizontal.svg',
  sidebar,
  basePath: '/docs',
  topNav: [
    { text: 'Book', link: '/getting-started/install' },
    { text: 'Specs', link: '/sdk' },
    {
      element: React.createElement('a', { href: '/docs', target: '_self' }, 'Rustdocs')
    },
    { text: 'GitHub', link: 'https://github.com/openvm-org/openvm' },
    {
      text: 'v1.4.0',
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
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
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