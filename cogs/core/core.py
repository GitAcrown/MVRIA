import io
import logging
import textwrap
import traceback
from contextlib import redirect_stdout
from typing import Any, Optional

import discord
from discord import app_commands
from discord.ext import commands

logger = logging.getLogger(f'MVRIA.{__name__.split(".")[-1]}')

class Core(commands.Cog):
    """Module central du bot, contenant des commandes de base."""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        
        self._last_result: Optional[Any] = None

    # Gestion des commandes et modules ------------------------------

    @commands.command(name="load", hidden=True)
    @commands.is_owner()
    async def load(self, ctx, *, cog: str):
        """Charge un module"""
        cog_path = f'cogs.{cog}.{cog}'
        try:
            await self.bot.load_extension(cog_path)
        except Exception as exc:
            await ctx.send(f"**`ERREUR :`** {type(exc).__name__} - {exc}")
        else:
            await ctx.send("**`SUCCÈS`**")

    @commands.command(name="unload", hidden=True)
    @commands.is_owner()
    async def unload(self, ctx, *, cog: str):
        """Décharge un module"""
        cog_path = f'cogs.{cog}.{cog}'
        try:
            await self.bot.unload_extension(cog_path)
        except Exception as exc:
            await ctx.send(f"**`ERREUR :`** {type(exc).__name__} - {exc}")
        else:
            await ctx.send("**`SUCCÈS`**")

    @commands.command(name="reload", hidden=True)
    @commands.is_owner()
    async def reload(self, ctx, *, cog: str):
        """Recharge un module"""
        cog_path = f'cogs.{cog}.{cog}'
        try:
            await self.bot.reload_extension(cog_path)
        except Exception as exc:
            await ctx.send(f"**`ERREUR :`** {type(exc).__name__} - {exc}")
        else:
            await ctx.send("**`SUCCÈS`**")
            
    @commands.command(name="reloadall", hidden=True)
    @commands.is_owner()
    async def reloadall(self, ctx):
        """Recharge tous les modules"""
        for ext_name, _ext in self.bot.extensions.items():
            try:
                await self.bot.reload_extension(ext_name)
            except Exception as exc:
                await ctx.send(f"**`ERREUR :`** {type(exc).__name__} - {exc}")
        await ctx.send("**`SUCCÈS`**")

    @commands.command(name="extensions", hidden=True)
    @commands.is_owner()
    async def extensions(self, ctx):
        for ext_name, _ext in self.bot.extensions.items():
            await ctx.send(ext_name)

    @commands.command(name="cogs", hidden=True)
    @commands.is_owner()
    async def cogs(self, ctx):
        for cog_name, _cog in self.bot.cogs.items():
            await ctx.send(cog_name)
            
    # Commandes d'évaluation de code ------------------------------
            
    def cleanup_code(self, content: str) -> str:
        # remove ```py\n```
        if content.startswith('```') and content.endswith('```'):
            return '\n'.join(content.split('\n')[1:-1])

        # remove `foo`
        return content.strip('` \n')
            
    @commands.command(name='eval', hidden=True)
    @commands.is_owner()
    async def eval_code(self, ctx: commands.Context, *, body: str):
        """Evalue du code"""

        env = {
            'bot': self.bot,
            'ctx': ctx,
            'channel': ctx.channel,
            'author': ctx.author,
            'guild': ctx.guild,
            'message': ctx.message,
            '_': self._last_result,
        }

        env.update(globals())

        body = self.cleanup_code(body)
        stdout = io.StringIO()

        to_compile = f'async def func():\n{textwrap.indent(body, "  ")}'

        try:
            exec(to_compile, env)
        except Exception as e:
            return await ctx.send(f'```py\n{e.__class__.__name__}: {e}\n```')

        func = env['func']
        try:
            with redirect_stdout(stdout):
                ret = await func()
        except Exception as e:
            value = stdout.getvalue()
            await ctx.send(f'```py\n{value}{traceback.format_exc()}\n```')
        else:
            value = stdout.getvalue()
            try:
                await ctx.message.add_reaction('\u2705')
            except:
                pass

            if ret is None:
                if value:
                    await ctx.send(f'```py\n{value}\n```')
            else:
                self._last_result = ret
                await ctx.send(f'```py\n{value}{ret}\n```')
                
    @app_commands.command(name="ping")
    async def ping(self, interaction: discord.Interaction) -> None:
        """Renvoie le ping du bot"""
        await interaction.response.send_message(f"Pong ! (`{round(self.bot.latency * 1000)}ms`)")

async def setup(bot):
    await bot.add_cog(Core(bot))
