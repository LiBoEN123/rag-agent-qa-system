import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import MarkdownContent from '../components/MarkdownContent.vue'

describe('assistant Markdown', () => {
  it('renders Markdown and removes executable markup, unsafe links, and external image loads', () => {
    const wrapper = mount(MarkdownContent, { props: { content: '**正常内容**\n<script>alert(1)</script>\n<img src=x onerror=alert(1)>\n[点击](javascript:alert(1))\n![追踪图](https://example.invalid/tracker)' } })
    expect(wrapper.find('strong').text()).toBe('正常内容')
    expect(wrapper.find('script').exists()).toBe(false)
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.find('[onerror]').exists()).toBe(false)
    expect(wrapper.find('a[href^="javascript:"]').exists()).toBe(false)
  })
})
