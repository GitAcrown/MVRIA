import asyncio
import io
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence
from googlesearch import search
import numexpr as ne
import requests
from bs4 import BeautifulSoup

import discord
import openai
import pytz
import tiktoken
import unidecode
import random
from discord import Interaction, app_commands
from discord.ext import commands
from moviepy import VideoFileClip
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from common import dataio
from common.utils import fuzzy

logger = logging.getLogger(f'MVRIA.{__name__.split(".")[-1]}')

COMPLETION_MODEL_GUILD = 'gpt-4o'
COMPLETION_MODEL_DM = 'gpt-4o-mini'
AUDIO_TRANSCRIPTION_MODEL = 'whisper-1'
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_COMPLETION_TOKENS = 500
DEFAULT_CONTEXT_WINDOW = 10000
DEFAULT_TOOLS_ENALBED = True
CONTEXT_CLEANUP_DELAY = timedelta(minutes=10)
WEB_CHUNK_SIZE = 1500

DEFAULT_CUSTOM_GUILD = "Réponds aux questions des utilisateurs de manière concise et simple en adaptant ton langage à celui de tes interlocuteurs."
DEFAULT_CUSTOM_DM = "Sois le plus direct et concis possible dans tes réponses. N'hésite pas à poser des questions pour mieux comprendre les besoins de l'utilisateur."

GUILD_DEVELOPER_PROMPT = lambda d: f'''[BEHAVIOR]
Tu es {d['assistant_name']}, un chatbot conversant avec des utilisateurs dans un salon textuel Discord.
Ne met jamais ton propre nom ou l'horodatage devant tes réponses.
Tu peux analyser les images qu'on te donne.
Tu dois suivre scrupuleusement les instructions personnalisées.
[INFO]
- Users messages format : '<name> <datetime> : <msg>'.
- Current date/time (ISO 8601): {d['current_datetime']}
- Weekday: {d['weekday']}
[TOOLS]
Tu es encouragé à utiliser plusieurs outils à la fois si nécessaire.
- REMINDERS: Tu peux créer des rappels QUE pour l'utilisateur demandeur lorsqu'il le demande. A proposer dès que ça peut être utile.
- WEB SEARCH: Tu peux rechercher des sites et lire leur contenu pour obtenir des informations. A utiliser pour répondre à des questions.
- EVALUATE MATH: Tu peux évaluer des expressions mathématiques de manière précise.
[CUSTOM INSTRUCTIONS]
{d['custom_instructions']}'''

DM_DEVELOPER_PROMPT = lambda d: f'''[BEHAVIOR]
Tu es {d['assistant_name']}, un assistant personnel ayant pour but d'aider ton utilisateur dans ses tâches quotidiennes.
Ne met jamais l'horodatage devant tes réponses.
Tu peux analyser les images que l'utilisateur te donne.
Tu dois suivre scrupuleusement les instructions personnalisées.
[INFO]
- User name: {d['user_name']}
- User messages format : '<datetime> : <msg>'
- Current date/time (ISO 8601): {d['current_datetime']}
- Weekday: {d['weekday']}
[TOOLS]
Tu es encouragé à utiliser plusieurs outils à la fois si nécessaire.
- REMINDERS: Tu peux créer des rappels pour l'utilisateur lorsqu'il le demande. A proposer dès que ça peut être utile.
- WEB SEARCH:  Tu peux rechercher des sites et lire leur contenu pour obtenir des informations. A utiliser pour répondre à des questions.
- EVALUATE MATH: Tu peux évaluer des expressions mathématiques de manière précise.
[CUSTOM INSTRUCTIONS]
{d['custom_instructions']}'''

# EXCEPTIONS -----------------------------------------------------------------

class GPTWrapperError(Exception):
    """Classe de base pour les exceptions du wrapper GPT4o"""
    pass

class OpenAIError(GPTWrapperError):
    """Classe de base pour les erreurs OpenAI"""
    pass

# UI -------------------------------------------------------------------------

class SystemPromptModal(discord.ui.Modal, title="Modifier le comportement"):
    """Modal pour modifier ou consulter le prompt du système."""
    def __init__(self, current_system_prompt: str, max_length: int = 500) -> None:
        super().__init__(timeout=None)
        self.current_system_prompt = current_system_prompt
        
        self.new_system_prompt = discord.ui.TextInput(
            label="Instructions",
            style=discord.TextStyle.long,
            placeholder="Instructions de comportement de l'assistant",
            default=self.current_system_prompt,
            required=True,
            min_length=10,
            max_length=max_length
        )
        self.add_item(self.new_system_prompt)
        
    async def on_submit(self, interaction: Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        return self.stop()
        
    async def on_error(self, interaction: Interaction, error: Exception) -> None:
        return await interaction.response.send_message(f"**Erreur** × {error}", ephemeral=True)

class ConfirmView(discord.ui.View):
    """Permet de confirmer une action."""
    def __init__(self, author: discord.Member | discord.User):
        super().__init__()
        self.value = False
        self.author = author
    
    @discord.ui.button(label="Confirmer", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        self.value = True
        self.stop()
    
    @discord.ui.button(label="Annuler", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        self.value = False
        self.stop()

# UTILS ------------------------------------------------------------------------
def sanitize_text(text: str) -> str:
    """Retire les caractères spéciaux d'un texte."""
    text = ''.join([c for c in unidecode.unidecode(text) if c.isalnum() or c.isspace()]).rstrip()
    return re.sub(r"[^a-zA-Z0-9_-]", "", text[:32])

# OUTILS -----------------------------------------------------------------------
class GPTTool:
    def __init__(self,
                 name: str,
                 description: str,
                 properties: dict,
                 function: Callable[['ToolCall', 'InteractionGroup'], 'ToolAnswerMessage'],
                 *,
                 footer: str = '') -> None:
        self.name = name
        self.description = description
        self.properties = properties
        self.function = function
        self.footer = footer
        
        self._required = [k for k, _ in properties.items()]
        
    def __repr__(self) -> str:
        return f"<GPTTool {self.name}>"
    
    def execute(self, tool_call: 'ToolCall', interaction: 'InteractionGroup') -> 'ToolAnswerMessage':
        return self.function(tool_call, interaction)
    
    @property
    def to_dict(self):
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'strict': True,
                'parameters': {
                    'type': 'object',
                    'required': self._required,
                    'properties': self.properties,
                    'additionalProperties': False
                }
            }
        }
        
class ToolCall:
    def __init__(self, **data):
        self.data = data
        
    def __repr__(self) -> str:
        return f"<ToolCall {self.data}>"
    
    @property
    def function_name(self) -> str:
        return self.data['function']['name']
    
    @property
    def arguments(self) -> dict:
        return json.loads(self.data['function']['arguments'])
    
    @classmethod
    def from_message_tool_call(cls, message_tool_call: ChatCompletionMessageToolCall) -> 'ToolCall':
        return cls(
            id=message_tool_call.id,
            type='function',
            function={
                'name': message_tool_call.function.name,
                'arguments': message_tool_call.function.arguments
            }
        )

# CHUNKS -----------------------------------------------------------------------
class MessageChunk:
    def __init__(self, **data):
        self.data = data
        
    def __repr__(self) -> str:
        return f"<MessageChunk {self.data}>"
    
    def __eq__(self, other):
        return self.data == other.data
    
    @property
    def type(self):
        return self.data.get('type', 'text')
    
    @property
    def token_count(self): 
        if self.data.get('token_count'):
            return self.data['token_count']
        tokenizer = tiktoken.get_encoding('cl100k_base')
        if self.type == 'text':
            return len(tokenizer.encode(self.data['text']))
        elif self.type == 'image_url':
            return 250 # Estimation vague
        return 0
       
    @classmethod
    def from_dict(cls, data: dict) -> 'MessageChunk':
        if data.get('type') == 'image_url':
            return ImageURLChunk(**data)
        return TextChunk(**data)
    
class TextChunk(MessageChunk):
    def __init__(self, text: str):
        super().__init__(type='text', text=text)
    
class ImageURLChunk(MessageChunk):
    def __init__(self, image_url: str, detail: Literal['low', 'high', 'auto'] = 'auto'):
        super().__init__(type='image_url', image_url={'url': image_url, 'detail': detail})
    
# MESSAGES ---------------------------------------------------------------------
class ContextMessage:
    def __init__(self,
                 role: Literal['user', 'assistant', 'developer', 'tool'],
                 content: str | Iterable[MessageChunk] | None,
                 *,
                 identifier: str | None = None,
                 timestamp: datetime = datetime.now(pytz.utc),
                 **kwargs):
        self.role = role
        self.identifier = sanitize_text(identifier) if identifier else None
        self.timestamp = timestamp
        
        self._content = content
        self._token_count : int = kwargs.get('token_count', 0)
        
    def __repr__(self) -> str:
        return f"<ContextMessage {self.role} {self._content}>"
    
    @property
    def content(self) -> Iterable[MessageChunk]:
        if isinstance(self._content, str):
            return [TextChunk(self._content)]
        elif isinstance(self._content, (list, tuple)):
            return self._content
        return []
    
    @property
    def token_count(self) -> int:
        if self._token_count:
            return self._token_count
        return sum(chunk.token_count for chunk in self.content)
    
    @property
    def is_empty(self) -> bool:
        return not any(self.content)
    
    @property
    def payload(self) -> dict:
        if self.identifier:
            return {
                'role': self.role,
                'content': [chunk.data for chunk in self.content],
                'name': self.identifier
            }
        return {
            'role': self.role,
            'content': [chunk.data for chunk in self.content]
        }
        
class DeveloperMessage(ContextMessage):
    def __init__(self, content: str | Iterable[MessageChunk] | None, **kwargs):
        super().__init__('developer', content, **kwargs)
        
class AssistantMessage(ContextMessage):
    def __init__(self,
                 content: str,
                 *,
                 timestamp: datetime = datetime.now(pytz.utc),
                 token_count: int = 0,
                 finish_reason: str | None = None,
                 **kwargs):
        super().__init__('assistant', content, timestamp=timestamp, token_count=token_count, **kwargs)
        self._content = content
        self.finish_reason = finish_reason
        
    @classmethod
    def from_chat_completion(cls, chat_completion: ChatCompletion) -> 'AssistantMessage':
        if not chat_completion.choices:
            raise ValueError('Completion has no choices')
        comp = chat_completion.choices[0]
        usage = chat_completion.usage.completion_tokens if chat_completion.usage else 0
        return cls(
            content=comp.message.content if comp.message.content else '[MESSAGE VIDE]',
            token_count=usage,
            finish_reason=comp.finish_reason
        )
        
class AssistantToolCalls(ContextMessage):
    def __init__(self,
                 tool_calls: Iterable[ToolCall]) -> None:
        super().__init__('assistant', None)
        self.tool_calls = tool_calls
        
    @property
    def payload(self) -> dict:
        return {
            'role': self.role,
            'tool_calls': [call.data for call in self.tool_calls]
        }
        
class ToolAnswerMessage(ContextMessage):
    def __init__(self,
                 content: dict,
                 tool_call_id: str,
                 **kwargs) -> None:
        super().__init__('tool', json.dumps(content), **kwargs)
        self.tool_call_id = tool_call_id
        
    @property
    def payload(self) -> dict:
        return {
            'role': self.role,
            'content': self._content,
            'tool_call_id': self.tool_call_id
        }
        
class UserMessage(ContextMessage):
    def __init__(self, 
                 content: str | Iterable[MessageChunk],
                 name: str,
                 discord_message: discord.Message,
                 *,
                 timestamp: datetime = datetime.now(pytz.utc)) -> None:
        super().__init__('user', content, identifier=name, timestamp=timestamp)
        self.discord_message = discord_message
        self._author : discord.User | discord.Member | None = discord_message.author if discord_message else None
        self._channel : discord.TextChannel | discord.Thread | None = discord_message.channel if discord_message else None #type: ignore
        
    @classmethod
    async def from_discord_message(cls, message: discord.Message) -> 'UserMessage':
        content = []
        guild = message.guild
        
        ref_message = message.reference.resolved if message.reference else None
        if message.content:
            author_name = message.author.name if not message.author.bot else f'{message.author.name}[BOT]'
            horodatage = message.created_at.astimezone(pytz.timezone('Europe/Paris')).isoformat()
            san_content = message.content.replace(guild.me.mention, '').strip() if guild else message.content
            if message.guild:
                msg_content = f'{author_name} {horodatage}:{san_content}'
            else:
                msg_content = f'{horodatage}:{san_content}'
            if isinstance(ref_message, discord.Message) and ref_message.content:
                ref_author_name = ref_message.author.name if not ref_message.author.bot else f'{ref_message.author.name}[BOT]'
                msg_content = f'[QUOTING:] {ref_author_name} {ref_message.created_at.astimezone(pytz.timezone("Europe/Paris")).isoformat()}:{ref_message.content}\n[MESSAGE:] {msg_content}'
            content.append(TextChunk(msg_content))
                
        image_urls = []
        for msg in [message, ref_message]:
            if not isinstance(msg, discord.Message):
                continue
            
            for embed in msg.embeds:
                embed_content = ''
                if embed.title:
                    embed_content += f'{embed.title}\n'
                if embed.description:
                    embed_content += f'{embed.description}\n'
                
                if embed_content:
                    if msg == message:
                        content.append(TextChunk(f'[FROM EMBED:] {embed_content}'))
                    else:
                        content.append(TextChunk(f'[QUOTING:][FROM EMBED:] {embed_content}'))
            
            # Images fournies 
            for attachment in msg.attachments:
                if attachment.content_type and attachment.content_type.startswith('image'):
                    image_urls.append(attachment.url)
            # URL d'images
            for match in re.finditer(r'(https?://[^\s]+)', msg.content):
                url = match.group(1)
                cleanurl = re.sub(r'\?.*$', '', url)
                if cleanurl.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_urls.append(url)
            # Images dans les embeds
            for embed in msg.embeds:
                if embed.image and embed.image.url:
                    image_urls.append(embed.image.url)
            # Stickers
            for sticker in msg.stickers:
                image_urls.append(sticker.url)
            
        if image_urls:
            content.extend([ImageURLChunk(url) for url in image_urls])
            
        return cls(content, name=message.author.name, discord_message=message, timestamp=message.created_at)
    
# GROUPES D'INTERACTIONS -------------------------------------------------------

class InteractionGroup:
    def __init__(self,
                 messages: list[ContextMessage],
                 **extras) -> None:
        self.messages = messages
        self.extras = extras
        
    def __repr__(self) -> str:
        return f"<InteractionGroup {len(self.messages)} messages>"
    
    def __eq__(self, other):
        return all(msg in other.messages for msg in self.messages)
    
    @property
    def total_token_count(self) -> int:
        return sum(msg.token_count for msg in self.messages)
    
    @property
    def is_empty(self) -> bool:
        return not any(self.messages)
    
    @property
    def completed(self) -> bool:
        return len(self.messages) > 1 and self.messages[-1].role == 'assistant'
    
    # Messages
    
    def check_messages(self, cond: Callable[[ContextMessage], bool]) -> bool:
        return any(cond(msg) for msg in self.messages)
    
    def get_messages(self, role: Literal['user', 'assistant', 'developer', 'tool']) -> list[ContextMessage]:
        return [msg for msg in self.messages if msg.role == role]
    
    def append_messages(self, *messages: ContextMessage) -> None:
        self.messages.extend(messages)
        
    def remove_message(self, message: ContextMessage) -> None:
        self.messages.remove(message)
        
    @property
    def last_message(self) -> ContextMessage:
        return self.messages[-1]
    
    @property
    def last_completion(self) -> AssistantMessage | None:
        return next((msg for msg in reversed(self.messages) if isinstance(msg, AssistantMessage)), None)
    
    # Utilitaires
    
    @property
    def contains_image(self) -> bool:
        return self.check_messages(lambda msg: any(isinstance(chunk, ImageURLChunk) for chunk in msg.content))
    
    @property
    def contains_tool_call(self) -> bool:
        return self.check_messages(lambda msg: msg.role == 'tool')
    
    def fetch_author(self) -> discord.User | discord.Member | None:
        user_message = self.get_messages('user')[0]
        if isinstance(user_message, UserMessage):
            return user_message._author
        return None
    
    def fetch_channel(self) -> discord.TextChannel | discord.Thread | None:
        user_message = self.get_messages('user')[0]
        if isinstance(user_message, UserMessage):
            return user_message._channel
        return None
    
# SESSIONS ---------------------------------------------------------------------

class AssistantSession:
    def __init__(self,
                 cog: 'Assistant',
                 channel: discord.TextChannel | discord.Thread | discord.DMChannel,
                 custom_instructions: str,
                 temperature: float = DEFAULT_TEMPERATURE,
                 tools: list[GPTTool] = [],
                 *,
                 max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
                 context_window: int = DEFAULT_CONTEXT_WINDOW,
                 tools_enabled: bool = DEFAULT_TOOLS_ENALBED,
                 session_user_name: str = '') -> None:
        self._cog = cog
        self.channel = channel
        self.custom_instructions = custom_instructions
        self.temperature = temperature
        self.tools = tools
        self.max_completion_tokens = max_completion_tokens
        self.context_window = context_window
        self.tools_enabled = tools_enabled
        
        self._session_user_name = session_user_name
        
        self._interactions : list[InteractionGroup] = []
        self._last_cleanup = datetime.now(pytz.utc)
        
        self._session_start = datetime.now(pytz.utc)
        
    def __repr__(self) -> str:
        return f"<AssistantSession {self.channel}>"
    
    @property
    def is_private(self) -> bool:
        return not self.channel.guild
    
    @property
    def gpt_model(self) -> str:
        return COMPLETION_MODEL_DM if self.is_private else COMPLETION_MODEL_GUILD
    
    @property
    def developer_prompt(self) -> DeveloperMessage:
        data = {
            'assistant_name': 'MARIA',
            'current_datetime': datetime.now().isoformat(),
            'weekday': datetime.now().strftime('%A'),
            'custom_instructions': self.custom_instructions,
            'user_name': self._session_user_name
        }
        return DeveloperMessage(GUILD_DEVELOPER_PROMPT(data) if not self.is_private else DM_DEVELOPER_PROMPT(data))
    
    # Gestion des messages
    
    def get_interaction(self, index: int) -> InteractionGroup | None:
        return self._interactions[index] if 0 <= index < len(self._interactions) else None
    
    def get_last_interaction(self) -> InteractionGroup | None:
        return self._interactions[-1] if self._interactions else None
    
    def retrieve_interaction(self, message_id: int) -> InteractionGroup | None:
        for interaction in self._interactions:
            if interaction.check_messages(lambda msg: msg.discord_message.id == message_id if isinstance(msg, UserMessage) else False):
                return interaction
    
    def create_interaction(self, *messages: ContextMessage, **extras) -> InteractionGroup:
        interaction = InteractionGroup(list(messages), **extras)
        self._interactions.append(interaction)
        return interaction
    
    def remove_interaction(self, interaction: InteractionGroup) -> None:
        self._interactions.remove(interaction)
        
    def clear_interactions(self, cond: Callable[[InteractionGroup], bool] = lambda i: True) -> None:
        self._interactions = [i for i in self._interactions if not cond(i)]
        
    def cleanup_interactions(self, older_than: timedelta):
        now = datetime.now(pytz.utc)
        if self._last_cleanup + CONTEXT_CLEANUP_DELAY < now:
            self.clear_interactions(lambda i: i.last_message.timestamp.astimezone(pytz.utc) < now - older_than)
            self._last_cleanup = now
            
    # Outils
    
    def get_tool(self, name: str) -> GPTTool :
        for t in self.tools:
            if t.name == name:
                return t
        raise ValueError(f'Aucun outil trouvé pour {name}')
            
    # Contexte
    
    def get_context(self) -> Sequence[ContextMessage]:
        ctx = []
        cleanup_age = timedelta(hours=4) if self.channel.guild else timedelta(hours=48)
        self.cleanup_interactions(cleanup_age)
        tokens = self.developer_prompt.token_count
        for interaction in self._interactions[::-1]:
            tokens += interaction.total_token_count
            if tokens > self.context_window:
                break
            ctx.append(interaction)
        return [self.developer_prompt] + [m for i in ctx[::-1] for m in i.messages]
    
    # Completion
    
    async def complete(self, current_interaction: InteractionGroup) -> InteractionGroup:
        messages = [m.payload for m in self.get_context()]
        
        current_interaction.extras['user'] = current_interaction.fetch_author()
        current_interaction.extras['tools_usage'] = current_interaction.extras.get('tools_usage', [])
        
        try:
            completion = await self._cog.client.chat.completions.create(
                model=self.gpt_model,
                messages=messages, #type: ignore
                max_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                tools=[t.to_dict for t in self.tools], #type: ignore
                parallel_tool_calls=True,
                timeout=30
            )
        except openai.BadRequestError as e:
            if 'invalid_image_url' in str(e):
                self.clear_interactions(lambda i: i.contains_image)
                return await self.complete(current_interaction)
            logger.error(e, exc_info=True)
            raise e
        
        try:
            assistant_message = AssistantMessage.from_chat_completion(completion)
        except ValueError as e:
            logger.error(e, exc_info=True)
            raise OpenAIError('Impossible de créer un message à partir de la complétion')
        
        comp = completion.choices[0]
        if self.tools_enabled and comp.message.tool_calls:
            tool_calls = [ToolCall.from_message_tool_call(call) for call in comp.message.tool_calls]
            current_interaction.append_messages(AssistantToolCalls(tool_calls))
            tool_answers = []
            for tool_call in tool_calls:
                tool = self.get_tool(tool_call.function_name)
                tool_answer = tool.execute(tool_call, current_interaction)
                if tool_answer:
                    tool_answers.append(tool_answer)
                    current_interaction.extras['tools_usage'].append(tool_call.function_name)
                else:
                    logger.warning(f'No answer for tool {tool_call.function_name}')
            if tool_answers:
                current_interaction.append_messages(*tool_answers)
                return await self.complete(current_interaction)
            
        if assistant_message.is_empty:
            if assistant_message.finish_reason == 'content_filter':
                assistant_message._content = "<:error_icon:1338657710333362198> **Contenu filtré par OpenAI** × Veuillez reformuler votre question."
            elif not current_interaction.extras.get('retry', False):
                current_interaction.extras['retry'] = True
                return await self.complete(current_interaction)
    
        current_interaction.append_messages(assistant_message)
        return current_interaction

class Assistant(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        # Paramètres de l'assistant sur les serveurs
        guild_config = dataio.DictTableBuilder(
            name='guild_config',
            default_values={
                'developer_prompt': DEFAULT_CUSTOM_GUILD,
                'temperature': DEFAULT_TEMPERATURE,
                'answer_to': f'{bot.user.name.lower()}' if bot.user else '',
                'authorized': True
            },
            insert_on_reconnect=True
        )
        self.data.map_builders(discord.Guild, guild_config)
        
        # Paramètres de l'assistant en messages privés
        user_config = dataio.TableBuilder(
            '''CREATE TABLE IF NOT EXISTS user_config (
                user_id INTEGER PRIMARY KEY,
                custom_instructions TEXT DEFAULT '',
                temperature REAL DEFAULT 0.9,
                authorized INTEGER DEFAULT -1
                )'''
        )
        
        self.data.map_builders('global', user_config)
        
        self.client = AsyncOpenAI(api_key=self.bot.config['OPENAI_API_KEY']) #type: ignore
        
        self.create_audio_transcription = app_commands.ContextMenu(
            name="Transcription audio",
            callback=self.transcript_audio_callback)
        self.bot.tree.add_command(self.create_audio_transcription)
        
        self._sessions = {}
        self._busy : dict[int, bool] = {}
        self.page_chunks_cache = {}
        
        self.GPT_TOOLS = [
            GPTTool(name='search_web_page',
                    description="Recherche des pages web et renvoie une description des pages trouvées.",
                    properties={'query': {'type': 'string', 'description': 'Requête de recherche'},
                                'num_results': {'type': 'number', 'description': 'Nombre de résultats à renvoyer (max. 5)'},
                                'lang': {'type': 'string', 'description': "Langue de recherche (ex. 'fr', 'en')"}},
                    function=self._tool_search_web_pages,
                    footer="<:websearch_icon:1340801019281670255> Recherche web"),
            GPTTool(name='navigate_page_chunks',
                    description="Navigue et renvoie une partie (chunk) d'une page web. A utiliser pour lire le contenu des pages trouvées avec l'outil de recherche web.",
                    properties={
                        'url': {'type': 'string', 'description': 'URL de la page dont on souhaite extraire le contenu'},
                        'index': {'type': ['number', 'null'], 'description': "Index de la partie (chunk) de la page à renvoyer (0-indexé)"},
                    },
                    function=self._tool_navigate_page_chunks,
                    footer="<:web_icon:1338659113638297670> Lecture de page web"),
            GPTTool(name='math_eval',
                   description="Évalue une expression mathématique. Utilise la syntaxe Python standard avec les opérateurs mathématiques classiques.",
                   properties={'expression': {'type': 'string', 'description': "L'expression mathématique à évaluer"}},
                   function=self._tool_math_eval,
                   footer="<:math_icon:1339332020458754161> Calcul mathématique"),
        ]
        
    async def cog_unload(self):
        self.data.close_all()
        
    # SESSIONS -----------------------------------------------------------------
    
    async def get_session(self, bucket: discord.TextChannel | discord.Thread | discord.User | discord.Member) -> AssistantSession | None:
        if bucket.id in self._sessions:
            return self._sessions[bucket.id]
        
        channel = None
        if isinstance(bucket, discord.User) or isinstance(bucket, discord.Member):
            custom_instructions = self.get_user_config(bucket).get('custom_instructions')
            if not custom_instructions:
                custom_instructions = DEFAULT_CUSTOM_DM
            temperature = float(self.get_user_config(bucket).get('temperature', DEFAULT_TEMPERATURE))
            if not bucket.dm_channel:
                try:
                    channel = await bucket.create_dm()
                except discord.Forbidden:
                    return None
        elif bucket.guild:
            custom_instructions = self.get_guild_config(bucket.guild)['developer_prompt']
            temperature = float(self.get_guild_config(bucket.guild)['temperature'])
            channel = bucket
        if not channel:
            return None
        
        self._check_cogs_for_tools()
        
        session = AssistantSession(self, 
                                   channel, 
                                   custom_instructions, 
                                   temperature, 
                                   self.GPT_TOOLS, 
                                   session_user_name=bucket.name if isinstance(bucket, (discord.User, discord.Member)) else '',
                                   context_window=int(DEFAULT_CONTEXT_WINDOW / 2) if isinstance(bucket, (discord.User, discord.Member)) else DEFAULT_CONTEXT_WINDOW)
        self._sessions[bucket.id] = session
        return session
    
    async def remove_session(self, bucket: discord.TextChannel | discord.Thread | discord.User | discord.Member) -> None:
        if bucket.id in self._sessions:
            del self._sessions[bucket.id]
        
    # Configurations -----------------------------------------------------------
    
    # Guild
    def get_guild_config(self, guild: discord.Guild) -> dict:
        return self.data.get(guild).get_dict_values('guild_config')
    
    def set_guild_config(self, guild: discord.Guild, **config):
        self.data.get(guild).set_dict_values('guild_config', {k: v for k, v in config.items() if k in ['developer_prompt', 'temperature', 'authorized', 'answer_to']})
        
    # User
    def get_user_config(self, user: discord.User | discord.Member) -> dict:
        r = self.data.get().fetchone('SELECT * FROM user_config WHERE user_id = ?', user.id)
        if not r:
            return {}
        return dict(r)
    
    def set_user_custom_instructions(self, user: discord.User | discord.Member, instructions: str):
        self.data.get().execute('INSERT OR REPLACE INTO user_config (user_id, custom_instructions) VALUES (?, ?)', user.id, instructions)
        
    def set_user_temperature(self, user: discord.User | discord.Member, temperature: float):
        self.data.get().execute('INSERT OR REPLACE INTO user_config (user_id, temperature) VALUES (?, ?)', user.id, temperature)
        
    def set_user_authorized(self, user: discord.User | discord.Member, authorized: int):
        """Définit si l'utilisateur est autorisé à utiliser l'assistant.

        :param user: Utilisateur à configurer
        :param authorized: -1 pour laisser la configuration du serveur, 0 pour interdire, 1 pour autoriser
        """
        if authorized not in (-1, 0, 1):
            raise ValueError('Valeur d\'autorisation invalide')
        self.data.get().execute('INSERT OR REPLACE INTO user_config (user_id, authorized) VALUES (?, ?)', user.id, authorized)
        
    # Réponse automatique
    def should_trigger_reply(self, guild: discord.Guild, message: discord.Message) -> bool:
        """Renvoie si l'assistant doit répondre à un message à la mention de son nom."""
        string = self.get_guild_config(guild)['answer_to']
        if not string:
            return False
        if re.search(rf'\b{re.escape(string)}\b[.!?]*?', message.content, re.IGNORECASE):
            return True
        return False
        
    # Autorisation d'utilisation
    def is_user_authorized(self, user: discord.User | discord.Member) -> bool:
        """Vérifie si l'utilisateur est autorisé à utiliser l'assistant
        Si l'utilisateur est membre d'un serveur autorisé, il est autorisé, sinon on vérifie sa configuration personnelle"""
        user_auth = self.get_user_config(user).get('authorized', -1)
        if user_auth == -1:
            mutual_guilds = [g for g in user.mutual_guilds if self.is_guild_authorized(g)]
            return bool(mutual_guilds)
        return bool(user_auth)
    
    def is_guild_authorized(self, guild: discord.Guild) -> bool:
        return self.get_guild_config(guild)['authorized']
    
    # Notes -------------------------------------------------------------------
    
    def fetch_user_from_name(self, guild: discord.Guild, name: str) -> discord.Member | None:
        user = discord.utils.find(lambda u: u.name == name.lower(), guild.members)
        if user:
            return user if user else None
        
        # On tente d'extraire un ID
        poss_id = re.search(r'\d{17,19}', name)
        if poss_id:
            user = discord.utils.find(lambda u: u.id == int(poss_id.group(0)), guild.members)
            return user if user else None
        
        # On cherche le membre le plus proche en nom
        members = [member.name for member in guild.members]
        closest_member = fuzzy.extract_one(name, members)
        if closest_member:
            user = discord.utils.find(lambda u: u.name == closest_member[0], guild.members)
            return user if user else None
        
        # On cherche le membre le plus proche en surnom
        nicknames = [member.nick for member in guild.members if member.nick]
        closest_nickname = fuzzy.extract_one(name, nicknames, score_cutoff=90)
        if closest_nickname:
            user = discord.utils.find(lambda u: u.nick == closest_nickname[0], guild.members)
            return user if user else None
    
    def internal_memory_get(self, title: str) -> dict | None:
        title = title.lower().strip()
        r = self.data.get().fetchone('SELECT * FROM internal_memory WHERE title = ?', title)
        return r if r else None
    
    def internal_memory_write(self, title: str, content: str, owner: discord.User | discord.Member | None = None, guild: discord.Guild | None = None) -> None:
        title = title.lower().strip()
        # On enregistre ou met à jour la note mais on ne change pas le verrouillage
        self.data.get().execute('INSERT OR REPLACE INTO internal_memory (title, content, lock_user, owner_id, lock_guild, guild_id) VALUES (?, ?, (SELECT lock_user FROM internal_memory WHERE title = ?), ?, (SELECT lock_guild FROM internal_memory WHERE title = ?), ?)', title, content, title, owner.id if owner else None, title, guild.id if guild else None)
        
    def internal_memory_lock(self, title: str, user: bool, guild: bool) -> None:
        title = title.lower().strip()
        self.data.get().execute('UPDATE internal_memory SET lock_user = ?, lock_guild = ? WHERE title = ?', user, guild, title)
        
    def internal_memory_delete(self, title: str) -> None:
        title = title.lower().strip()
        self.data.get().execute('DELETE FROM internal_memory WHERE title = ?', title)
        
    def internal_memory_list_titles(self, user: discord.User | discord.Member | None, guild: discord.Guild | None) -> list[str]:
        if user:
            r = self.data.get().fetchall('SELECT title FROM internal_memory WHERE owner_id = ?', user.id)
        elif guild:
            r = self.data.get().fetchall('SELECT title FROM internal_memory WHERE guild_id = ?', guild.id)
        else:
            r = self.data.get().fetchall('SELECT title FROM internal_memory WHERE owner_id IS NULL AND guild_id IS NULL')
        return [row[0] for row in r]
    
    def internal_memory_search(self, pattern: str) -> list[str]:
        r = self.data.get().fetchall('SELECT title FROM internal_memory WHERE title LIKE ? OR content LIKE ? SORT BY last_update DESC', f'%{pattern}%', f'%{pattern}%')
        return [row[0] for row in r]
    
    # Google ------------------------------------------------------------
    
    def search_web_pages(self, query: str, lang: str = 'fr', num_results: int = 3):
        """Recherche des informations sur le web."""
        results = search(query, lang=lang, num_results=num_results, advanced=True, safe='off')
        return results

    def fetch_page_chunks(self, url: str, chunk_size: int = WEB_CHUNK_SIZE) -> list[str]:
        if url in self.page_chunks_cache and self.page_chunks_cache[url]['chunk_size'] == chunk_size:
            return self.page_chunks_cache[url]['chunks']
        
        # Récupération avec un user-agent pour éviter les blocages
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return []
        except Exception:
            return []
            
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Suppression agressive des éléments non pertinents
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "iframe", 
                        "noscript", "form", "button", "svg", "canvas", ".ad", ".ads", ".cookie", 
                        ".popup", ".banner", ".sidebar", ".menu", ".comments"]):
            tag.decompose()
        
        # Détection du contenu principal
        main_content = None
        for selector in ['main', 'article', '.content', '.post', '.post-content', '.entry-content', 
                        '#content', '#main', '.article', '[role="main"]']:
            content = soup.select(selector)
            if content and len(str(content[0])) > 500:
                main_content = content[0]
                break
        
        # Utiliser le contenu principal ou le corps si non trouvé
        if main_content:
            text_container = main_content
        else:
            body = soup.find('body')
            text_container = body if body else soup
        
        # Extraction intelligente du texte pertinent
        text = ""
        for elem in text_container.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th']):
            content = elem.get_text(strip=True)
            # Ignorer les éléments courts ou vides
            if len(content) > 20 or (elem.name.startswith('h') and content):
                text += elem.name.startswith('h') and f"\n## {content}\n" or f"\n{content}\n"
        
        # Nettoyage du texte
        text = re.sub(r'\n{3,}', '\n\n', text)  # Réduire les sauts de ligne excessifs
        text = re.sub(r' {2,}', ' ', text)      # Réduire les espaces multiples
        
        # Supprimer les phrases communes de RGPD, cookies, etc.
        noise_patterns = [
            r'En poursuivant votre navigation.*?cookies',
            r'This site uses cookies.*?privacy',
            r'Nous utilisons des cookies.*?confidentialité',
            r'Accept [aA]ll cookies',
            r'cookie policy',
            r'privacy policy',
            r'terms of service'
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Suppression des lignes qui semblent être des URLs isolées
        text = re.sub(r'^(https?:|www\.).*$', '', text, flags=re.MULTILINE)
        
        # Nettoyage final des caractères non imprimables et normalisation des espaces
        text = ''.join(c for c in text if c.isprintable() or c in "\n")
        text = text.strip()
        
        # Division intelligente en chunks qui respecte les paragraphes
        chunks = []
        paragraphs = [p for p in re.split(r'\n\n+', text) if p.strip()]
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si l'ajout du paragraphe dépasse la taille du chunk
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk = f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
        
        # Ajouter le dernier chunk s'il n'est pas vide
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Nettoyer les chunks vides et mettre en cache
        chunks = [c for c in chunks if len(c.strip()) > 100]
        self.page_chunks_cache[url] = {'chunks': chunks, 'chunk_size': chunk_size}
        return chunks
    
    # OUTILS -------------------------------------------------------------------
    
    def _check_cogs_for_tools(self):
        # Recherche sur les autres cogs si y'a une liste GPT_TOOLS pour les ajouter
        for cog in self.bot.cogs.values():
            if cog.qualified_name == self.qualified_name:
                continue
            if hasattr(cog, 'GPT_TOOLS'):
                added = self._add_tools(*cog.GPT_TOOLS) #type: ignore
                if added:
                    logger.info(f"i --- Outils ajoutés depuis '{cog.qualified_name}'")
    
    def _add_tools(self, *tools: GPTTool) -> list[GPTTool]:
        added = []
        for tool in tools:
            if tool.name not in (t.name for t in self.GPT_TOOLS):
                self.GPT_TOOLS.append(tool)
                added.append(tool)
        return added
        
    def _remove_tools(self, tools_names: list[str]):
        self.GPT_TOOLS = [t for t in self.GPT_TOOLS if t.name not in tools_names]
    
    def _tool_search_web_pages(self, tool_call: ToolCall, interaction: InteractionGroup) -> ToolAnswerMessage:
        query = tool_call.arguments.get('query')
        num = tool_call.arguments.get('num_results', 5)
        lang = tool_call.arguments.get('lang', 'fr')
        if not query:
            return ToolAnswerMessage({'error': 'Argument manquant'}, tool_call.data['id'])
        results = []
        for r in self.search_web_pages(query, lang, num):
            results.append({'title': r.title, 'url': r.url, 'description': r.description}) #type: ignore
        return ToolAnswerMessage({'results': results}, tool_call.data['id'])
    
    def _tool_math_eval(self, tool_call: ToolCall, interaction: InteractionGroup) -> ToolAnswerMessage:
        """Évalue une expression mathématique de manière sécurisée"""
        expr = tool_call.arguments.get('expression', '').strip()
        if not expr:
            return ToolAnswerMessage({'error': 'Expression manquante'}, tool_call.data['id'])
        
        try:
            # On évalue l'expression de manière sécurisée avec numexpr
            result = float(ne.evaluate(expr))
            if result.is_integer():
                result = int(result)
            return ToolAnswerMessage({'result': result, 'expression': expr}, tool_call.data['id'])
        except Exception as e:
            return ToolAnswerMessage({'error': f"Erreur d'évaluation : {str(e)}"}, tool_call.data['id'])

    def _tool_navigate_page_chunks(self, tool_call: ToolCall, interaction: InteractionGroup) -> ToolAnswerMessage:
        url = tool_call.arguments.get('url')
        if not url:
            return ToolAnswerMessage({'error': "Aucune URL fournie."}, tool_call.data['id'])
        # Récupérer l'index ou utiliser 0 par défaut
        index = tool_call.arguments.get('index')
        try:
            idx = int(index) if index is not None else 0
        except Exception:
            return ToolAnswerMessage({'error': "Index non valide."}, tool_call.data['id'])
        
        # Utiliser le cache (ou le remplir) pour obtenir les chunks
        chunks = self.page_chunks_cache.get(url, {}).get('chunks')
        if not chunks:
            chunks = self.fetch_page_chunks(url)
            if not chunks:
                return ToolAnswerMessage({'error': "Impossible de récupérer le contenu de la page."}, tool_call.data['id'])
        if idx < 0 or idx >= len(chunks):
            return ToolAnswerMessage({'error': "Index hors limites.", 'chunks_total': len(chunks)}, tool_call.data['id'])
        return ToolAnswerMessage({'chunk': chunks[idx], 'index': idx, 'chunks_total': len(chunks)}, tool_call.data['id'])
    
    # AUDIO --------------------------------------------------------------------
    
    async def extract_message_audio(self, message: discord.Message) -> io.BytesIO | Path | None:
        for attachment in message.attachments:
            # Message audio
            if attachment.content_type and attachment.content_type.startswith('audio'):
                buffer = io.BytesIO()
                buffer.name = attachment.filename
                await attachment.save(buffer, seek_begin=True)
                return buffer
            # Vidéo
            elif attachment.content_type and attachment.content_type.startswith('video'):
                path = self.data.get_subfolder('temp', create=True) / f'{datetime.now().timestamp()}.mp4'
                await attachment.save(path)
                clip = VideoFileClip(str(path))
                audio = clip.audio
                if not audio:
                    return None
                audio_path = path.with_suffix('.wav')
                audio.write_audiofile(str(audio_path))
                clip.close()
                
                os.remove(str(path))
                return audio_path
        return None
    
    async def audio_transcription(self, file: io.BytesIO | Path, model: str = AUDIO_TRANSCRIPTION_MODEL) -> str:
        try:
            transcript = await self.client.audio.transcriptions.create(
                model=model,
                file=file,
            )
        except Exception as e:
            logger.error(e, exc_info=True)
            raise OpenAIError('Impossible de transcrire l\'audio')
        if isinstance(file, io.BytesIO):
            file.close()
        return transcript.text
    
    async def transcript_audio_callback(self, interaction: Interaction, message: discord.Message):
        if interaction.channel_id != message.channel.id:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Le message doit être dans le même salon", ephemeral=True)
        if not message.attachments:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur** × Aucun fichier n'est attaché au message.", ephemeral=True)
        
        await interaction.response.send_message("**Récupération de l'audio...**")
        file = await self.extract_message_audio(message)
        if not file:
            return await interaction.edit_original_response(content="<:error_icon:1338657710333362198> **Erreur** × Aucun audio n'a pu être extrait du message.")
        
        await interaction.edit_original_response(content="**Transcription en cours...**")
        try:
            transcript = await self.audio_transcription(file)
        except OpenAIError as e:
            return await interaction.edit_original_response(content=f"<:error_icon:1338657710333362198> **Erreur** × {e}")
        
        if type(file) is Path:
            file.unlink()
        await interaction.delete_original_response()
        
        transcript = f'>>> {transcript}\n-# <:transcript_icon:1338656808918712331> Transcription demandée par {interaction.user.mention}'
        
        content = []
        if len(transcript) >= 2000:
            content = [transcript[i:i+2000] for i in range(0, len(transcript), 2000)]
        else:
            content = [transcript]
        for _, chunk in enumerate(content):
            await message.reply(chunk, mention_author=False)
            
    # LISTENERS ----------------------------------------------------------------
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if not isinstance(message.channel, (discord.TextChannel, discord.Thread, discord.DMChannel)):
            return
        if message.mention_everyone:
            return
        
        bucket = message.channel if message.guild else message.author
        horodatage = message.created_at.astimezone(pytz.timezone('Europe/Paris')).isoformat()
        is_transcript = False
        # Session privée (DM) -----------------------------------------------
        if isinstance(bucket, (discord.User, discord.Member)):
            if not self.is_user_authorized(bucket):
                return await message.channel.send("<:error_icon:1338657710333362198> **Action impossible** × Vous n'êtes pas autorisé à utiliser l'assistant.")
            if self._busy.get(bucket.id, False):
                return await message.channel.send("<:timer_icon:1338660569921622107> **Trop rapide !** × Attendez que la requête précédente soit traitée...", delete_after=5)
            self._busy[bucket.id] = True
            
            session = await self.get_session(bucket)
            if not session:
                return await message.reply("<:error_icon:1338657710333362198> **Action impossible** × Impossible de créer une session privée.")
            
            user_message = await UserMessage.from_discord_message(message)
            if not user_message.content:
                # Si y'a un message audio, on le transcrit et l'incorpore au message
                if message.attachments:
                    audio = await self.extract_message_audio(message)
                    if audio:
                        try:
                            transcription = await self.audio_transcription(audio)
                        except OpenAIError as e:
                            return await message.reply(f"<:error_icon:1338657710333362198> **Erreur** × {e}", mention_author=False)
                        is_transcript = True
                        user_message = UserMessage([TextChunk(f'{horodatage}: [FROM AUDIO TRANSCRIPTION:] {transcription}')], name=message.author.name, discord_message=message)
                if not user_message.content:
                    return
                
        # Session serveur (seulement si bot mentionné) ----------------------
        elif isinstance(bucket, (discord.TextChannel, discord.Thread)):
            if bucket.guild.me.mentioned_in(message) or self.should_trigger_reply(bucket.guild, message):
                
                if not self.is_guild_authorized(bucket.guild):
                    return await message.channel.send("<:error_icon:1338657710333362198> **Action impossible** × Le serveur n'est pas autorisé à utiliser l'assistant.")
                
                session = await self.get_session(bucket)
                if not session:
                    return await message.channel.send("<:error_icon:1338657710333362198> **Action impossible** × Impossible de créer une session sur le serveur.")
                
                user_message = await UserMessage.from_discord_message(message)
                if not user_message.content:
                    if message.attachments:
                        audio = await self.extract_message_audio(message)
                        if audio:
                            try:
                                transcription = await self.audio_transcription(audio)
                            except OpenAIError as e:
                                return await message.reply(f"<:error_icon:1338657710333362198> **Erreur** × {e}", mention_author=False)
                            is_transcript = True
                            user_message = UserMessage([TextChunk(f'{message.author.name} {horodatage}: [FROM AUDIO TRANSCRIPTION:] {transcription}')], name=message.author.name, discord_message=message)
            else:
                return
        else:
            return
            
        interaction = session.create_interaction(user_message)    
        async with message.channel.typing():
            try:
                interaction = await session.complete(interaction)
            except OpenAIError as e:
                logger.error(e, exc_info=True)
                return await message.reply("<:error_icon:1338657710333362198> **Erreur** × Impossible de compléter la requête.\nVeuillez réessayer plus tard ou réinitialiser la session avec `/reboot`.", mention_author=False)
            
            completion = interaction.last_completion
            if not completion:
                return await message.reply("<:error_icon:1338657710333362198> **Erreur** × Aucune réponse n'a été générée pour cette requête.", mention_author=False)
            
            text = completion._content
            tools_usage = interaction.extras.get('tools_usage', [])
            # Compter l'utilisation de chaque outil
            tools_count = {}
            for tool_name in tools_usage:
                tools_count[tool_name] = tools_count.get(tool_name, 0) + 1
                
            footers = []
            # Ajouter les footers avec leur compte d'utilisation
            for tool in self.GPT_TOOLS:
                if tool.name in tools_count:
                    count = tools_count[tool.name]
                    footer = tool.footer + (f' (x{count})' if count > 1 else '')
                    footers.append(footer)
                    
            if is_transcript:
                footers.append("<:transcript_icon:1338656808918712331> Transcription audio")
            if footers:
                text += '\n-# ' + ' · '.join((f for f in footers if f))
            
            content = [text[i:i+2000] for i in range(0, len(text), 2000)]
            for _, chunk in enumerate(content):
                if isinstance(bucket, (discord.User, discord.Member)):
                    await message.channel.send(chunk, mention_author=False)
                else:
                    await message.reply(chunk, mention_author=False)
                
        self._busy[bucket.id] = False
                
    @commands.Cog.listener()
    async def on_message_delete(self, message: discord.Message):
        if message.author.bot:
            return
        if not isinstance(message.channel, (discord.TextChannel, discord.Thread)):
            return
        
        bucket = message.channel if message.guild else message.author
        session = await self.get_session(bucket)
        if not session:
            return
        
        interaction = session.retrieve_interaction(message.id)
        if not interaction:
            return
        
        session.remove_interaction(interaction)
        
    # COMMANDES =================================================================
    
    @app_commands.command(name='behavior')
    async def cmd_custom_instructions(self, interaction: Interaction):
        """Permet de personaliser le comportement de l'assistant."""
        bucket = interaction.channel if interaction.guild else interaction.user
        if isinstance(bucket, (discord.User, discord.Member)):
            if not self.is_user_authorized(bucket):
                return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Vous n'êtes pas autorisé à utiliser l'assistant.", ephemeral=True)
        elif isinstance(bucket, (discord.TextChannel, discord.Thread)):
            if not self.is_guild_authorized(bucket.guild):
                return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Le serveur n'est pas autorisé à utiliser l'assistant.", ephemeral=True)
        else:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Impossible de créer une session.", ephemeral=True)
        
        session = await self.get_session(bucket)
        if not session:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Session invalide** × Impossible de créer une session.", ephemeral=True)
        
        prompt = SystemPromptModal(session.custom_instructions, max_length=200 if session.is_private else 500)
        await interaction.response.send_modal(prompt)
        if await prompt.wait():
            return await interaction.followup.send("**Action annulée** × Délai de réponse dépassé.", ephemeral=True)
        new_prompt = prompt.new_system_prompt.value
        if new_prompt == session.custom_instructions:
            return await interaction.followup.send("**Information** × Les instructions de l'assistant n'ont pas été modifiées.", ephemeral=True)
        
        session.custom_instructions = new_prompt
        if isinstance(bucket, (discord.User, discord.Member)):
            self.set_user_custom_instructions(bucket, new_prompt)
        elif isinstance(bucket, (discord.TextChannel, discord.Thread)):
            self.set_guild_config(bucket.guild, developer_prompt=new_prompt)
            
        await interaction.followup.send(f"## <:settings_icon:1338659554921156640> __Instructions de comportement modifiées :__\n>>> *{new_prompt}*\n-# Il est conseillé d'effectuer un `/reboot` afin d'éviter un conflit avec les anciennes instructions.", ephemeral=False)
        
    @app_commands.command(name='temperature')
    @app_commands.rename(temp='température')
    async def cmd_temperature(self, interaction: Interaction, temp: app_commands.Range[float, 0.0, 2.0]):
        """Modifier le degré de créativité de l'assistant.

        :param temp: Température de génération, entre 0.0 et 2.0"""
        bucket = interaction.channel if interaction.guild else interaction.user
        if isinstance(bucket, (discord.User, discord.Member)):
            if not self.is_user_authorized(bucket):
                return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Vous n'êtes pas autorisé à utiliser l'assistant.", ephemeral=True)
        elif isinstance(bucket, (discord.TextChannel, discord.Thread)):
            if not self.is_guild_authorized(bucket.guild):
                return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Le serveur n'est pas autorisé à utiliser l'assistant.", ephemeral=True)
        else:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Impossible de créer une session.", ephemeral=True)
        
        session = await self.get_session(bucket)
        if not session:
            return await interaction.response.send_message("**Session invalide** × Impossible de créer une session.", ephemeral=True)
    
        # On met à jour la température
        session.temperature = temp
        if isinstance(bucket, (discord.User, discord.Member)):
            self.set_user_temperature(bucket, temp)
        elif isinstance(bucket, (discord.TextChannel, discord.Thread)):
            self.set_guild_config(bucket.guild, temperature=temp)
            
        if temp > 1.4:
            return await interaction.response.send_message(f"<:settings_icon:1338659554921156640> **Température mise à jour** · La température de génération est désormais à ***{temp}***.\n-# Attention, une température élevée peut entraîner des réponses incohérentes.")
        await interaction.response.send_message(f"<:settings_icon:1338659554921156640> **Température mise à jour** · La température de génération est désormais à ***{temp}***.")
        
    @app_commands.command(name='mention')
    @app_commands.guild_only()
    @app_commands.rename(name='nom')
    async def cmd_mention(self, interaction: Interaction, name: app_commands.Range[str, 1, 32] = ''):
        """Indiquer à l'assistant s'il doit répondre à une mention indirecte du nom configuré.

        :param name: Nom à mentionner pour déclencher une réponse (laisser vide pour désactiver)"""
        if not interaction.guild:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Cette commande est réservée aux serveurs.", ephemeral=True)
        
        if not name:
            self.set_guild_config(interaction.guild, answer_to='')
            return await interaction.response.send_message("<:settings_icon:1338659554921156640> **Mention désactivée** · L'assistant ne répondra plus à une mention indirecte de son nom.\n-# L'assistant continuera à répondre aux mentions directes.")
        
        if not name.isascii() or re.search(r'\s', name):
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Nom invalide** × Le nom doit être en caractères ASCII et ne doit pas contenir d'espaces.", ephemeral=True)
        
        self.set_guild_config(interaction.guild, answer_to=name.lower())
        await interaction.response.send_message(f"<:settings_icon:1338659554921156640> **Mention configurée** · L'assistant répondra à une mention indirecte de ***{name}*** en plus des mentions directes.")
        
    @app_commands.command(name='info')
    async def cmd_info(self, interaction: Interaction):
        """Afficher les informations sur l'assistant sur la session en cours."""
        bucket = interaction.channel if interaction.guild else interaction.user
        if not isinstance(bucket, (discord.User, discord.Member, discord.TextChannel, discord.Thread)):
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Impossible de créer une session.", ephemeral=True)
        
        session = await self.get_session(bucket)
        if not session:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        embed = discord.Embed(title="<:settings_icon:1338659554921156640> Assistant · Infos sur la session", color=discord.Color(0x000001))
        embed.set_thumbnail(url=self.bot.user.display_avatar.url if self.bot.user else None)
        embed.set_footer(text="Implémentation de GPT4o-mini et Whisper-1 (par OpenAI)", icon_url="https://static-00.iconduck.com/assets.00/openai-icon-2021x2048-4rpe5x7n.png")
        
        # Informations sur l'assistant
        embed.add_field(name="Instructions", value=f"```{session.custom_instructions}```")
        embed.add_field(name="Température", value=f"```{session.temperature}```")
        
        # Informations sur la session
        embed.add_field(name="Type | Modèle", value=f"```{'Privée' if session.is_private else 'Serveur'} | {session.gpt_model}```")
        embed.add_field(name="Active depuis", value=f"```{session._session_start.strftime('%d/%m/%Y à %H:%M:%S')}```")
        embed.add_field(name="Nb. d'interactions", value=f"```{len(session._interactions)}```")
        embed.add_field(name="Tokens en mémoire", value=f"```{sum(i.total_token_count for i in session._interactions)}/{session.context_window}```")
        
        await interaction.response.send_message(embed=embed)
        
    @app_commands.command(name='reboot')
    async def cmd_reboot(self, interaction: Interaction):
        """Réinitialiser la mémoire de la session actuelle de l'assistant."""
        bucket = interaction.channel if interaction.guild else interaction.user
        if not isinstance(bucket, (discord.User, discord.Member, discord.TextChannel, discord.Thread)):
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Impossible de créer une session.", ephemeral=True)
        
        session = await self.get_session(bucket)
        if not session:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        session.clear_interactions()
        session._last_cleanup = datetime.now(pytz.utc)
        await interaction.response.send_message("<:settings_icon:1338659554921156640> **Session réinitialisée** × La mémoire de la session en cours de l'assistant a été réinitialisée.")
        
    @app_commands.command(name='factoryreset')
    async def cmd_factoryreset(self, interaction: Interaction):
        """Réinitialiser les paramètres de l'assistant."""
        bucket = interaction.channel if interaction.guild else interaction.user
        if not isinstance(bucket, (discord.User, discord.Member, discord.TextChannel, discord.Thread)):
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Action impossible** × Impossible de créer une session.", ephemeral=True)
        
        session = await self.get_session(bucket)
        if not session:
            return await interaction.response.send_message("<:error_icon:1338657710333362198> **Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        view = ConfirmView(interaction.user)
        await interaction.response.send_message(f"**Confirmation** · Êtes-vous sûr de vouloir réinitialiser les paramètres de l'assistant ?", ephemeral=True, view=view)
        await view.wait()
        if not view.value:
            return await interaction.edit_original_response(content="**Action annulée** · Les paramètres de l'assistant n'ont pas été réinitialisés.", view=None)
        
        if isinstance(bucket, (discord.User, discord.Member)):
            self.set_user_custom_instructions(bucket, DEFAULT_CUSTOM_DM)
            self.set_user_temperature(bucket, DEFAULT_TEMPERATURE)
        elif isinstance(bucket, (discord.TextChannel, discord.Thread)):
            self.set_guild_config(bucket.guild, developer_prompt=DEFAULT_CUSTOM_GUILD, temperature=DEFAULT_TEMPERATURE)
            
        session.custom_instructions = DEFAULT_CUSTOM_DM if session.is_private else DEFAULT_CUSTOM_GUILD
        session.temperature = DEFAULT_TEMPERATURE
        session.clear_interactions()
        await interaction.edit_original_response(content="<:settings_icon:1338659554921156640> **Paramètres réinitialisés** · Les paramètres de l'assistant ont été réinitialisés.", view=None)
        
    # COMMANDES OWNER ==========================================================
    
    @commands.command(name='guildauth')
    @commands.is_owner()
    async def cmd_guildauth(self, ctx: commands.Context, guild_id: int, value: bool):
        """Autoriser ou interdire un serveur à utiliser l'assistant."""
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return await ctx.send(f"**Erreur** × Serveur introuvable.")
        
        self.set_guild_config(guild, authorized=value)
        await ctx.send(f"**Autorisation d'utilisation SERVEUR** · Le serveur `{guild.name}` (ainsi que ses membres) est désormais {'autorisé' if value else 'interdit'} à utiliser l'assistant.")
        
    @commands.command(name='userauth')
    @commands.is_owner()
    async def cmd_userauth(self, ctx: commands.Context, user_id: int, value: int):
        """Autoriser ou interdire un utilisateur à utiliser l'assistant."""
        user = self.bot.get_user(user_id)
        if not user:
            return await ctx.send(f"**Erreur** × Utilisateur introuvable.")
        
        if value not in [-1, 0, 1]:
            return await ctx.send(f"**Erreur** × La valeur doit être -1 (défaut), 0 (interdit) ou 1 (autorisé).")
        
        self.set_user_authorized(user, value)
        text = "autorisé" if value == 1 else "interdit" if value == 0 else "défaut"
        await ctx.send(f"**Autorisation d'utilisation UTILISATEUR** · L'utilisateur `{user.name}` est désormais '{text}' à utiliser l'assistant.")
        
async def setup(bot):
    await bot.add_cog(Assistant(bot))
